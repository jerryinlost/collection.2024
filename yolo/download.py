from huggingface_hub import hf_hub_download

repos=\
[
    # 'keremberke/yolov8m-nlf-head-detection',
    'keremberke/yolov8n-nlf-head-detection',
    # 'keremberke/yolov8s-nlf-head-detection',
    # 'foduucom/product-detection-in-shelf-yolov8',
    # 'keremberke/yolov8s-blood-cell-detection',
    # 'keremberke/yolov8m-blood-cell-detection',
    'keremberke/yolov8n-blood-cell-detection',
    'keremberke/yolov8n-hard-hat-detection',
    # 'keremberke/yolov8m-table-extraction',
    'keremberke/yolov8n-table-extraction',
    'keremberke/yolov8n-protective-equipment-detection',
    # 'keremberke/yolov8m-protective-equipment-detection',
]
for repo in repos:
    repo_config = dict(
        repo_id = repo,
        filename = "best.pt",
        local_dir = f"./models/{repo}"
    )
    hf_hub_download(**repo_config)