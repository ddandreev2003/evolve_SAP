
import torch
import numpy as np
from diffusers import Flux2KleinPipeline
from typing import Any, Callable, Dict, List, Optional
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps, Flux2PipelineOutput
from diffusers.utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

def map_SAP_dict(pf_prompts, num_inference_steps):
    prompts_list = pf_prompts['prompts_list']
    switch_prompts_steps = pf_prompts['switch_prompts_steps']
    verify_SAP_prompts(prompts_list, switch_prompts_steps, num_inference_steps)
    SAP_mapping = {}
    prompt_index = 0
    for i in range(num_inference_steps):
        # If current step exceeds the next switch step, increment the prompt index
        if prompt_index < len(switch_prompts_steps) and i >= switch_prompts_steps[prompt_index]:
            prompt_index += 1
        SAP_mapping[f"step{i}"] = prompt_index


    return prompts_list, SAP_mapping

def verify_SAP_prompts(prompts_list, switch_prompts_steps, num_inference_steps):
    if len(prompts_list) < 1:
        raise ValueError(
                f"prompts_list is empty"
            )
    if len(prompts_list) !=  (len(switch_prompts_steps) +1):
        raise ValueError(
                f"len(prompts_list) !=  (len(switch_prompts_steps) +1). len(prompts_list): {len(prompts_list)}, len(switch_prompts_steps)+1: {(len(switch_prompts_steps) +1)}"
            )
    if len(switch_prompts_steps) > 0:
        if sorted(switch_prompts_steps) != switch_prompts_steps:
            raise ValueError(
                    f"switch_prompts_steps is not ordered. switch_prompts_steps: {switch_prompts_steps}"
                )
        if switch_prompts_steps[0] < 0 or switch_prompts_steps[-1] > num_inference_steps:
            raise ValueError(
                    f"switch_prompts_steps is out of boundes. switch_prompts_steps: {switch_prompts_steps}"
                )

class SapFlux(Flux2KleinPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image=None,
        sap_prompts=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator | List[torch.Generator]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds=None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
    ):
        if sap_prompts is None:
            raise ValueError("sap_prompts must be provided with prompts_list and switch_prompts_steps.")

        prompts_list, SAP_mapping = map_SAP_dict(sap_prompts, num_inference_steps)
        base_prompt = prompts_list[0]

        # 1. Check inputs
        self.check_inputs(
            prompt=base_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device

        # 3. Prepare prompt embeddings per SAP stage
        prompt_embeds_dicts = []
        for prompt in prompts_list:
            stage_prompt_embeds, stage_text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
            prompt_embeds_dicts.append({"prompt_embeds": stage_prompt_embeds, "text_ids": stage_text_ids})

        first_prompt_embeds = prompt_embeds_dicts[0]["prompt_embeds"]

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. Prepare latent variables
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=first_prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                # use corresponding proxy prompt embeds
                prompt_dict = prompt_embeds_dicts[SAP_mapping[f'step{i}']]
                prompt_embeds = prompt_dict["prompt_embeds"]
                text_ids = prompt_dict["text_ids"]

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1) :]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = self._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)