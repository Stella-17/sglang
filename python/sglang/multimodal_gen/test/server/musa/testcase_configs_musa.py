from __future__ import annotations

from dataclasses import replace

from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    MULTI_FRAME_I2I_sampling_params,
    MULTI_IMAGE_TI2I_sampling_params,
    T2I_sampling_params,
    T2V_sampling_params,
    TI2I_sampling_params,
    TI2V_sampling_params,
)

MUSA_TI2I_sampling_params = replace(
    TI2I_sampling_params,
    image_path="/hf-cache/hub/musa-test-assets/TI2I_Qwen_Image_Edit_Input.jpg",
)

ONE_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "qwen_image_t2i_musa",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image",
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_musa",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            modality="video",
            custom_validator="video",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
        run_consistency_check=False,
    ),
]


NIGHTLY_1_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "zimage_image_t2i_musa",
        DiffusionServerArgs(
            model_path="Tongyi-MAI/Z-Image-Turbo",
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_layered_i2i_musa",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image-Layered",
            modality="image",
        ),
        MULTI_FRAME_I2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "fast_hunyuan_video_musa",
        DiffusionServerArgs(
            model_path="FastVideo/FastHunyuan-diffusers",
            modality="video",
            custom_validator="video",
        ),
        T2V_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_2512_t2i_musa",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image-2512",
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_edit_t2i_musa",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image-Edit",
            modality="image",
        ),
        MUSA_TI2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_edit_2509_ti2i_musa",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image-Edit-2509",
            modality="image",
        ),
        MULTI_IMAGE_TI2I_sampling_params,
        run_consistency_check=False,
    ),
]


ONE_GPU_NIGHTLY_MUSA_CASES: list[DiffusionTestCase] = (
    ONE_GPU_MUSA_CASES + NIGHTLY_1_GPU_MUSA_CASES
)


TWO_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "wan2_1_i2v_14b_480P_2gpu_musa",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            modality="video",
            custom_validator="video",
            num_gpus=2,
        ),
        TI2V_sampling_params,
        run_consistency_check=False,
    ),
]
