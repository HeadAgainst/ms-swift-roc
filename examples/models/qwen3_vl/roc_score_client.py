from openai import OpenAI


def main():
    client = OpenAI(
        api_key='EMPTY',
        base_url='http://127.0.0.1:8000/v1',
    )
    model = client.models.list().data[0].id
    image_path = '/path/to/test.jpg'
    user_profile = '偏好主体清晰、色彩鲜艳、构图稳定的图片，不喜欢杂乱和灰暗的内容。'
    prompt = (
        'Task: rate_image_for_user\n'
        f'User profile: {user_profile}\n'
        'Please rate this image for this user.'
    )

    resp = client.chat.completions.create(
        model=model,
        max_tokens=256,
        temperature=0,
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': image_path
                },
                {
                    'type': 'text',
                    'text': prompt
                },
            ]
        }])
    print(resp.choices[0].message.content)


if __name__ == '__main__':
    main()
