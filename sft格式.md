**训练模型的评分能力格式：**

```python
  {
    "gt_score": gt_score,
    "messages": [
      {
        "content": "<image>Task: rate_image_for_user\nUser profile: 偏好主体清晰、色彩鲜艳、构图稳定的图片，不喜欢杂乱和灰暗的内容。\nPlease rate this image for this user.",
        "role": "user"
      },
      {
        "content": "<reason>reason</reason> <score>score</score>",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg"
    ]
  }
```

**训练模型总结用户profile的能力：**

```python
{
  "messages": [
    {
      "role": "user",
      "content": "<image><image><image>Task: summarize_user_profile\nBased on the user's ratings and these images, summarize the user profile.\nImage 1: score=5\nImage 2: score=2\nImage 3: score=4"
    },
    {
      "role": "assistant",
      "content": "user_profile"
    }
  ],
  "images": [
    "mllm_demo_data/1.jpg",
    "mllm_demo_data/2.jpg",
    "mllm_demo_data/3.jpg"
  ]
}
```