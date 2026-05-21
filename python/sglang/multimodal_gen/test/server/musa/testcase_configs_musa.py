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
            # model_path="Qwen/Qwen-Image",
            model_path="/hf-cache/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6",
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_musa",
        DiffusionServerArgs(
            # model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            model_path="/hf-cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b",
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
            model_path="/hf-cache/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6",
            # model_path="/data/models/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6",
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_layered_i2i_musa",
        DiffusionServerArgs(
            model_path="/hf-cache/hub/models--Qwen--Qwen-Image-Layered/snapshots/8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf",
            # model_path="/data/models/hub/models--Qwen--Qwen-Image-Layered/snapshots/8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf",
            modality="image",
        ),
        MULTI_FRAME_I2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "fast_hunyuan_video_musa",
        DiffusionServerArgs(
            model_path="/hf-cache/hub/models--FastVideo--FastHunyuan-diffusers/snapshots/7e948fca38562e218ae34485e005956592d36d9b",
            # model_path="/data/models/hub/models--FastVideo--FastHunyuan-diffusers/snapshots/7e948fca38562e218ae34485e005956592d36d9b",
            modality="video",
            custom_validator="video",
        ),
        T2V_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_2512_t2i_musa",
        DiffusionServerArgs(
            model_path="/hf-cache/hub/models--Qwen--Qwen-Image-2512/snapshots/25468b98e3276ca6700de15c6628e51b7de54a26",
            # model_path="/data/models/hub/models--Qwen--Qwen-Image-2512/snapshots/25468b98e3276ca6700de15c6628e51b7de54a26",
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_edit_t2i_musa",
        DiffusionServerArgs(
            model_path="/hf-cache/hub/models--Qwen--Qwen-Image-Edit/snapshots/ac7f9318f633fc4b5778c59367c8128225f1e3de",
            # model_path="/data/models/hub/models--Qwen--Qwen-Image-Edit/snapshots/ac7f9318f633fc4b5778c59367c8128225f1e3de",
            modality="image",
        ),
        MUSA_TI2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_edit_2509_ti2i_musa",
        DiffusionServerArgs(
            model_path="/hf-cache/hub/models--Qwen--Qwen-Image-Edit-2509/snapshots/d3968ef930e841f4c73640fb8afa3b306a78167e",
            # model_path="/data/models/hub/models--Qwen--Qwen-Image-Edit-2509/snapshots/d3968ef930e841f4c73640fb8afa3b306a78167e",
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
            # model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            model_path="/hf-cache/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/b184e23a8a16b20f108f727c902e769e873ffc73",
            modality="video",
            custom_validator="video",
            num_gpus=2,
        ),
        TI2V_sampling_params,
        run_consistency_check=False,
    ),
]
