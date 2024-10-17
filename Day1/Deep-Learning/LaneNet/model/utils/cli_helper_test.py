import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Image path")  # 이미지 파일 경로
    parser.add_argument("--video", help="Video path")  # 비디오 파일 경로
    parser.add_argument("--webcam", action='store_true', help="Use webcam for real-time lane detection")  # 웹캠 사용 옵션
    parser.add_argument("--model_type", help="Model type", default='DeepLabv3+')
    parser.add_argument("--model", help="Model path", default='./log/lanenet_DeepLabv3+_Focal_epoch100_batchsize8.pth')
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--save", help="Directory to save output", default="./test_output")
    return parser.parse_args()
