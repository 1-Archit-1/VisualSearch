import os
import cv2
import time
import random
import torch
import torch.nn.functional as F
import csv
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from youtubesearchpython import VideosSearch
import yt_dlp

# ==========================================
# DEVICE CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# CONFIGURATION
# ==========================================
REFERENCE_IMAGE_PATH = "image.png"  # Make sure this matches exactly

# Queries people eating lemons
QUERIES = [
    "i ate a whole lemon", "ate a whole lemon", "eating a whole lemon", 
    "eat a whole lemon", "me eating a whole lemon", "i ate an entire lemon", 
    "ate an entire lemon", "eating an entire lemon", "me eating an entire lemon", 
    "i tried eating a whole lemon", "trying to eat a whole lemon", 
    "i ate a raw lemon", "ate a raw lemon", "eating a raw lemon", 
    "eat a raw lemon", "me eating a raw lemon", "i ate raw lemons", 
    "eating raw lemons", "ate raw lemons", "i ate a lemon", "ate a lemon", 
    "eating a lemon", "eat a lemon", "me eating a lemon", "i tried eating a lemon", 
    "trying to eat a lemon", "i ate lemons", "eating lemons", "ate lemons"
]

MAX_RESULTS_PER_QUERY = 15  
FRAME_SKIP = 15             
MAX_SECONDS_TO_SCAN = 120   
OUTPUT_FILE = "top_100_lemon_v2.csv"
FRAMES_DIR = "best_lemon_frames"  

os.makedirs(FRAMES_DIR, exist_ok=True)

# ==========================================
# LOAD AI MODELS
# ==========================================
print(f"Loading CLIP model... This might take a minute.")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ==========================================
# PRE-CALCULATE REFERENCE IMAGE
# ==========================================
# ==========================================
# PRE-CALCULATE REFERENCE IMAGE
# ==========================================
print(f"Loading reference image: {REFERENCE_IMAGE_PATH}")
if not os.path.exists(REFERENCE_IMAGE_PATH):
    print(f"CRITICAL ERROR: Could not find {REFERENCE_IMAGE_PATH}! Please check the filename.")
    exit()

ref_image = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
ref_inputs = processor(images=ref_image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    ref_outputs = model.get_image_features(pixel_values=ref_inputs['pixel_values'])
    
    # Forcefully extract the tensor if Transformers wrapped it in an object
    if not isinstance(ref_outputs, torch.Tensor):
        if hasattr(ref_outputs, 'image_embeds'):
            ref_features = ref_outputs.image_embeds
        elif hasattr(ref_outputs, 'pooler_output'):
            ref_features = ref_outputs.pooler_output
        else:
            ref_features = ref_outputs[0]
    else:
        ref_features = ref_outputs
        
    ref_features = F.normalize(ref_features, p=2, dim=-1)

# ==========================================
# 1. SEARCH AND GATHER URLs
# ==========================================
def gather_video_urls(queries, max_results):
    print("\nGathering candidate videos...")
    unique_urls = set()
    df1 = pd.read_csv('top_100_lemon_videos.csv')
    df2 = pd.read_csv('top_100_lemon_videos2.csv')
    existing_urls= set(df1['URL'].tolist() + df2['URL'].tolist() )
    
#    # 1. Load existing URLs into a fast lookup set to skip them
#    existing_urls = set()
#    if os.path.exists(OUTPUT_FILE):
#        try:
#            df = pd.read_csv(OUTPUT_FILE)
#            existing_urls = set(df['URL'].tolist()) 
#            print(f"Skipping {len(existing_urls)} previously processed URLs...")
#        except Exception as e:
#            print(f"Could not read CSV (starting fresh): {e}")
#
    for query in queries:
        search_string = f"{query} before:2014"
        try:
            videos_search = VideosSearch(search_string, limit=max_results)
            results = videos_search.result()['result']
            for res in results:
                video_url = res['link']
                if video_url not in existing_urls:
                    unique_urls.add(video_url)
            time.sleep(random.uniform(1.5, 3.5)) 
        except Exception as e:
            print(f"Error searching {search_string}: {e}")
            
    print(f"Found {len(unique_urls)} NEW unique videos to analyze.")
    return list(unique_urls)

# ==========================================
# 2. DOWNLOAD LOW-RES VIDEO
# ==========================================
def download_lowest_quality_video(url, output_filename="temp_video.mp4"):
    ydl_opts = {
        'format': 'worstvideo[ext=mp4]/worst', 
        'outtmpl': output_filename,
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'sleep_interval': random.uniform(3, 7), 
        'max_sleep_interval': 10,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_filename
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

# ==========================================
# 3. EXTRACT AND SCORE FRAMES
# ==========================================
def score_video_with_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    max_score = 0.0
    best_frame_time = 0
    best_frame_image = None
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    max_frames_to_read = int(fps * MAX_SECONDS_TO_SCAN)

    frames_to_process = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count > max_frames_to_read: 
            break
        
        if count % FRAME_SKIP == 0:
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            frames_to_process.append(pil_image)
            timestamps.append(count / fps)
            
        count += 1
    cap.release()

    if not frames_to_process:
        return 0, 0, None

    batch_size = 16
    for i in range(0, len(frames_to_process), batch_size):
        batch_images = frames_to_process[i:i+batch_size]
        batch_times = timestamps[i:i+batch_size]
        
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            batch_outputs = model.get_image_features(pixel_values=inputs['pixel_values'])
            
            # Forcefully extract the tensor for the video frames too
            if not isinstance(batch_outputs, torch.Tensor):
                if hasattr(batch_outputs, 'image_embeds'):
                    batch_features = batch_outputs.image_embeds
                elif hasattr(batch_outputs, 'pooler_output'):
                    batch_features = batch_outputs.pooler_output
                else:
                    batch_features = batch_outputs[0]
            else:
                batch_features = batch_outputs
                
            batch_features = F.normalize(batch_features, p=2, dim=-1)
            
            # Cosine Similarity x 100 for a clean percentage-like score
            similarity = (ref_features @ batch_features.T).squeeze(0) * 100
            scores = similarity.cpu().numpy()
            
        if scores.ndim == 0:
            scores = [scores]

        for j, score in enumerate(scores):
            if float(score) > max_score:
                max_score = float(score)
                best_frame_time = batch_times[j]
                best_frame_image = batch_images[j]

    return max_score, best_frame_time, best_frame_image

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    urls = gather_video_urls(QUERIES, MAX_RESULTS_PER_QUERY)
    
    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            results = df.to_dict('records')
        except Exception:
            pass

    temp_file = "temp_video.mp4"

    for i, url in enumerate(urls):
        print(f"\nProcessing [{i+1}/{len(urls)}]: {url}")
        
        video_path = download_lowest_quality_video(url, temp_file)
        if not video_path or not os.path.exists(video_path):
            continue
            
        print("Running Image-to-Image analysis...")
        score, timestamp, best_image = score_video_with_clip(video_path)
        
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        timestamp_str = f"{mins}:{secs:02d}"
        
        print(f"Match Score: {score:.2f}/100 (Best match at {timestamp_str})")
        
        video_id = url.split("v=")[-1].split("&")[0][:11]
        
        if best_image:
            image_filename = f"score_{score:.1f}_{video_id}.jpg"
            image_path = os.path.join(FRAMES_DIR, image_filename)
            best_image.save(image_path)
            print(f"Saved thumbnail to {image_path}")

        results.append({
            "Rank": 0, 
            "Score": score,
            "Timestamp": timestamp_str,
            "URL": url
        })
        
        if os.path.exists(video_path):
            os.remove(video_path)

    # ==========================================
    # FINAL RANKING & EXPORT
    # ==========================================
    if not results:
        print("\nNo results to save.")
        return

    results.sort(key=lambda x: x["Score"], reverse=True)
    
    for rank, res in enumerate(results, 1):
        res["Rank"] = rank

    print(f"\nWriting {len(results)} results to {OUTPUT_FILE}...")
    keys = ["Rank", "Score", "Timestamp", "URL"]
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "="*50)
    print(f"✅ DONE! Saved to {OUTPUT_FILE}")
    print(f"📸 Check the '{FRAMES_DIR}' folder for the visual thumbnails!")
    print("="*50)

if __name__ == "__main__":
    main()
