#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import time
from torchvision import transforms
from PIL import Image as PILImage
import numpy as np
from model.lanenet.LaneNet import LaneNet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import threading

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class LaneDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.resize_height = 480
        self.resize_width = 640
        self.roi_height = self.resize_height // 3  # ROI 높이 설정

        # LaneNet 모델 로드
        self.model = LaneNet(arch='DeepLabv3+')
        self.model.load_state_dict(torch.load('/home/navigator/test_ws/src/erp42_control_ob/scripts/log/lanenet_DeepLabv3+_Focal_epoch100_batchsize8.pth'))
        self.model.eval()
        self.model.to(DEVICE)

        self.data_transform = transforms.Compose([
            transforms.Resize((self.roi_height, self.resize_width)),  # ROI 크기로 조정
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher("/lane_detection/output", Image, queue_size=1)
        self.lane_info_pub = rospy.Publisher("/lane_detection/lane_info", Float32MultiArray, queue_size=1)

        self.vehicle_center_x = self.resize_width // 2

        self.lane_colors = {
            'left': (255, 0, 0),    # 빨강
            'center': (0, 255, 0),  # 초록
            'right': (0, 0, 255)    # 파랑
        }

        self.processing_lock = threading.Lock()
        self.current_frame = None
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def load_test_data(self, img):
        img = PILImage.fromarray(img)
        img = self.data_transform(img)
        return img

    def _morphological_process(self, image, kernel_size=5):
        image = (image * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closing

    def _connect_components_analysis(self, image):
        return cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)

    def cluster_embeddings(self, binary_seg, instance_seg):
        idxs = np.where(binary_seg > 0.5)
        if len(idxs[0]) == 0:  # 조건을 만족하는 픽셀이 없는 경우
            rospy.logwarn("No pixels found above threshold in binary segmentation")
            return {}  # 빈 딕셔너리 반환

        embeddings = instance_seg[:, idxs[0], idxs[1]].transpose(1, 0)

        if embeddings.shape[0] < 2:  # DBSCAN에는 최소 2개의 샘플이 필요합니다
            rospy.logwarn("Not enough samples for clustering")
            return {}

        try:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)

            db = DBSCAN(eps=0.5, min_samples=min(100, embeddings.shape[0] // 2))  # min_samples 동적 조정
            db.fit(embeddings)

            lanes = defaultdict(list)
            for idx, label in enumerate(db.labels_):
                if label != -1:
                    lanes[label].append((idxs[0][idx], idxs[1][idx]))

            return lanes
        except Exception as e:
            rospy.logerr(f"Error in clustering: {e}")
            return {}

    def assign_lane_positions(self, lanes):
        if not lanes:
            return {}

        # 차선의 평균 x 좌표를 계산
        lane_avg_x = {lane_id: np.mean([p[1] for p in points]) for lane_id, points in lanes.items()}
       
        # x 좌표를 기준으로 차선을 정렬
        sorted_lanes = sorted(lane_avg_x.items(), key=lambda x: x[1])
       
        lane_positions = {}
        if len(sorted_lanes) == 1:
            lane_positions[sorted_lanes[0][0]] = 'center'
        elif len(sorted_lanes) == 2:
            lane_positions[sorted_lanes[0][0]] = 'left'
            lane_positions[sorted_lanes[1][0]] = 'right'
        elif len(sorted_lanes) >= 3:
            lane_positions[sorted_lanes[0][0]] = 'left'
            lane_positions[sorted_lanes[1][0]] = 'center'
            lane_positions[sorted_lanes[2][0]] = 'right'
       
        return lane_positions

    def process_frame(self, frame):
        # ROI 추출
        roi = frame[-self.roi_height:, :]
        input_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_img = PILImage.fromarray(input_img)
        input_np = np.array(input_img)

        dummy_input = self.load_test_data(input_np).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)

        with torch.no_grad():
            outputs = self.model(dummy_input)

        binary_seg = torch.squeeze(outputs['binary_seg_pred']).cpu().numpy()
        instance_seg = torch.squeeze(outputs['instance_seg_logits']).cpu().numpy()

        binary_seg = (binary_seg - binary_seg.min()) / (binary_seg.max() - binary_seg.min() + 1e-8)
        binary_seg = self._morphological_process(binary_seg)

        num_labels, labels, stats, centroids = self._connect_components_analysis(binary_seg)

        for index in range(1, num_labels):
            if stats[index][4] <= 100:
                binary_seg[labels == index] = 0

        binary_seg = binary_seg.astype(float) / 255.0

        lanes = self.cluster_embeddings(binary_seg, instance_seg)
        lane_positions = self.assign_lane_positions(lanes)

        result_img = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)

        if lanes:
            for lane_id, points in lanes.items():
                position = lane_positions.get(lane_id, 'center')  # 기본값은 'center'
                color = self.lane_colors[position]
                for point in points:
                    cv2.circle(result_img, (point[1], point[0]), 1, color, -1)

            lane_info = Float32MultiArray()
            lane_info.data = [item for sublist in lanes.values() for point in sublist for item in point]
            self.lane_info_pub.publish(lane_info)

            rospy.loginfo(f"Published lane info with {len(lanes)} lanes")
        else:
            rospy.logwarn("No lanes detected in this frame")

        cv2.line(result_img, (self.vehicle_center_x, 0),
                 (self.vehicle_center_x, self.roi_height), (255, 255, 255), 2)  # 차량 중앙선을 흰색으로 변경

        return result_img

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.processing_lock:
                self.current_frame = cv_image
        except Exception as e:
            rospy.logerr(e)

    def process_frames(self):
        while not rospy.is_shutdown():
            with self.processing_lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()

            start_time = time.time()
            result_img = self.process_frame(frame)

            # 원본 이미지에 결과 합성
            full_result = frame.copy()
            full_result[-self.roi_height:, :] = result_img

            result_msg = self.bridge.cv2_to_imgmsg(full_result, "bgr8")
            self.image_pub.publish(result_msg)

            cv2.imshow('Lane Detection with Vehicle Center', full_result)
            cv2.waitKey(1)

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            rospy.loginfo(f"FPS: {fps:.2f}")

def main():
    rospy.init_node('lane_detector', anonymous=False)
    lane_detector = LaneDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()