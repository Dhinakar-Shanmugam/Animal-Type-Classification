from icrawler.builtin import BingImageCrawler
import os

# Create folders if not exist
os.makedirs("dataset/cattle", exist_ok=True)
os.makedirs("dataset/buffalo", exist_ok=True)

# Keywords for cattle
cattle_keywords = [
    "cow farm side view",
    "cattle standing full body",
    "cow full body side view",
    "indian cow farm animal",
    "cow grazing field",
    "cow front view",
    "cow rear view",
    "cow village farm",
    "white cow indian",
    "brown cow farm"
]

# Keywords for buffalo
buffalo_keywords = [
    "buffalo indian farm animal",
    "buffalo side view",
    "buffalo full body side view",
    "black buffalo farm",
    "buffalo grazing field",
    "buffalo front view",
    "buffalo rear view",
    "buffalo village farm",
    "water buffalo india",
    "buffalo standing mud"
]

# Download cattle images
for i, keyword in enumerate(cattle_keywords):
    print(f"Downloading cattle images for: {keyword}")
    crawler = BingImageCrawler(storage={'root_dir': 'dataset/cattle'})
    crawler.crawl(
        keyword=keyword,
        max_num=100,
        file_idx_offset=i * 100
    )

# Download buffalo images
for i, keyword in enumerate(buffalo_keywords):
    print(f"Downloading buffalo images for: {keyword}")
    crawler = BingImageCrawler(storage={'root_dir': 'dataset/buffalo'})
    crawler.crawl(
        keyword=keyword,
        max_num=100,
        file_idx_offset=i * 100
    )

print("✅ Download completed!")