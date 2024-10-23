import os
import shutil
import numpy as np
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, HfFileSystem
from transformers import AutoProcessor
from datasets import load_dataset

MAX_LENGTH = 256
dataset = load_dataset("ShareGPT4Video/ShareGPT4Video", cache_dir='./cache/')
MODEL_ID = "llava-hf/LLaVa-NeXT-Video-7b-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False, cache_dir='./cache_processor/')
processor.tokenizer.padding_side = "right"

def read_video_decord(video_path, num_frames=16):
    '''
    Decode the video with Decord decoder.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly. Defaults to NUM_FRAMES

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    vr = VideoReader(uri=video_path, ctx=cpu(0))
    indices = np.arange(0, len(vr), len(vr) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames

def collate_fn(example, path, processor):
    """
    Function of data collator
    """
    video_file = example["video_path"].split("/")[-1]
    video_clip = read_video_decord(f'{path}/{video_file}')

    # overall caption to summarize all scene well
    captions_all = [caption for caption in example['captions'] if caption['idx'] == '-1']
    caption = captions_all[0]['content']

    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Provide a detailed summary for this video."},
                    {"type": "video"},
                    ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption},
                     ],
            },
        ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

    batch = processor(
        text=prompt,
        videos=video_clip,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    return batch


def prepare_small_dataset():
    datasets_combined = []
    fs = HfFileSystem()
    DATASET_PATH = "./dataset/"
    directory = f"{DATASET_PATH}/temp_dir"
    # ego4d-32.2G, mixit-22.2G, bdd100k-12.1G, pixabay - 20.1G, pexels - 14.3G
    zip_folders = { "mixit", "pexels" }

    downloaded_cache = {}

    for zip_folder in zip_folders:
        print(f"Working with {zip_folder}")
        zip_files = fs.ls(f"datasets/ShareGPT4Video/ShareGPT4Video/zip_folder/{zip_folder}", detail=False)
        
        for zip_file in zip_files:
            zip_file = zip_file.split("/")[-1]

            if zip_file in downloaded_cache:
                print(f"File {zip_file} is already downloaded, skipping...")
                path = downloaded_cache[zip_file]

            else:
                print(f"{zip_file} is downloading")
                path = hf_hub_download(
                    repo_id='ShareGPT4Video/ShareGPT4Video',
                    repo_type="dataset",
                    filename=f"zip_folder/{zip_folder}/{zip_file}",
                    local_dir=f"{DATASET_PATH}/{zip_folder}",
                    cache_dir=DATASET_PATH,
                )
                downloaded_cache[zip_file] = path  # Add to cache dictionary

            subdataset_name = zip_file.split("_")[0]

            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

            if path.endswith(".zip"):
                shutil.unpack_archive(path, directory)
                print(f"combining {zip_folder} to one df")
                curr_video_files = os.listdir(directory)
                small_dataset = dataset.filter(lambda example: example["video_path"].split("/")[-1] in curr_video_files)

                small_dataset = small_dataset.map(
                    collate_fn,
                    batched=False,
                    fn_kwargs={"path": directory},
                    num_proc=24,
                    remove_columns=["captions", "keyframe", "timestamp", "video_id", "video_path"],
                    writer_batch_size=500,
                )
                datasets_combined.append(small_dataset['train'])
                print(f"finished for {zip_folder}")