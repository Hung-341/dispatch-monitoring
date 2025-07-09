"""
Unit tests for utils module
"""
import pytest
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

# Import the functions to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    generate_colors,
    convert_bbox_to_z,
    convert_x_to_bbox,
    iou,
    associate_detections_to_trackers,
    KalmanBoxTracker,
    Sort,
    crop_bbox
)


class TestGenerateColors:
    """Test color generation functionality"""
    
    def test_generate_colors_positive(self):
        """Test generating positive number of colors"""
        n = 5
        colors = generate_colors(n)
        
        assert len(colors) == n
        assert all(isinstance(color, tuple) for color in colors)
        assert all(len(color) == 3 for color in colors)
        assert all(0 <= c <= 255 for color in colors for c in color)
    
    def test_generate_colors_zero(self):
        """Test generating zero colors"""
        colors = generate_colors(0)
        assert len(colors) == 0
    
    def test_generate_colors_large(self):
        """Test generating large number of colors"""
        n = 100
        colors = generate_colors(n)
        assert len(colors) == n


class TestBboxConversion:
    """Test bounding box conversion functions"""
    
    def test_convert_bbox_to_z(self):
        """Test converting bbox to state vector"""
        bbox = np.array([10, 20, 30, 40], dtype=np.float64)
        z = convert_bbox_to_z(bbox)
        
        assert z.shape == (4, 1)
        assert z[0, 0] == 20  # center x
        assert z[1, 0] == 30  # center y
        assert z[2, 0] == 200  # area (20 * 10)
        assert z[3, 0] == 2.0  # aspect ratio (20/10)
    
    def test_convert_x_to_bbox_no_score(self):
        """Test converting state vector to bbox without score"""
        x = np.array([[20, 30, 200, 2.0]], dtype=np.float64).T
        bbox = convert_x_to_bbox(x)
        
        assert bbox.shape == (1, 4)
        assert bbox[0, 0] == 10  # x1
        assert bbox[0, 1] == 20  # y1
        assert bbox[0, 2] == 30  # x2
        assert bbox[0, 3] == 40  # y2
    
    def test_convert_x_to_bbox_with_score(self):
        """Test converting state vector to bbox with score"""
        x = np.array([[20, 30, 200, 2.0]], dtype=np.float64).T
        score = 0.8
        bbox = convert_x_to_bbox(x, score)
        
        assert bbox.shape == (1, 5)
        assert bbox[0, 4] == score


class TestIoU:
    """Test Intersection over Union calculations"""
    
    def test_iou_identical_boxes(self):
        """Test IoU of identical bounding boxes"""
        bbox1 = np.array([10, 10, 30, 30], dtype=np.float64)
        bbox2 = np.array([10, 10, 30, 30], dtype=np.float64)
        
        iou_value = iou(bbox1, bbox2)
        assert iou_value == 1.0
    
    def test_iou_no_overlap(self):
        """Test IoU of non-overlapping bounding boxes"""
        bbox1 = np.array([10, 10, 20, 20], dtype=np.float64)
        bbox2 = np.array([30, 30, 40, 40], dtype=np.float64)
        
        iou_value = iou(bbox1, bbox2)
        assert iou_value == 0.0
    
    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping bounding boxes"""
        bbox1 = np.array([10, 10, 30, 30], dtype=np.float64)
        bbox2 = np.array([20, 20, 40, 40], dtype=np.float64)
        
        iou_value = iou(bbox1, bbox2)
        assert 0.0 < iou_value < 1.0
    
    def test_iou_contained_box(self):
        """Test IoU when one box is contained within another"""
        bbox1 = np.array([10, 10, 50, 50], dtype=np.float64)
        bbox2 = np.array([20, 20, 40, 40], dtype=np.float64)
        
        iou_value = iou(bbox1, bbox2)
        assert 0.0 < iou_value < 1.0


class TestAssociation:
    """Test detection-tracker association"""
    
    def test_associate_empty_trackers(self):
        """Test association with empty trackers"""
        detections = np.array([[10, 10, 20, 20, 0.8]], dtype=np.float64)
        trackers = np.empty((0, 5), dtype=np.float64)
        
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trackers
        )
        
        assert len(matches) == 0
        assert len(unmatched_dets) == 1
        assert len(unmatched_trks) == 0
    
    def test_associate_empty_detections(self):
        """Test association with empty detections"""
        detections = np.empty((0, 5), dtype=np.float64)
        trackers = np.array([[10, 10, 20, 20, 0.8]], dtype=np.float64)
        
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trackers
        )
        
        assert len(matches) == 0
        assert len(unmatched_dets) == 0
        assert len(unmatched_trks) == 1
    
    def test_associate_matching_boxes(self):
        """Test association with matching bounding boxes"""
        detections = np.array([[10, 10, 20, 20, 0.8]], dtype=np.float64)
        trackers = np.array([[10, 10, 20, 20, 0.8]], dtype=np.float64)
        
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trackers, iou_threshold=0.5
        )
        
        assert len(matches) == 1
        assert len(unmatched_dets) == 0
        assert len(unmatched_trks) == 0


class TestKalmanBoxTracker:
    """Test Kalman Box Tracker"""
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        bbox = np.array([10, 10, 20, 20], dtype=np.float64)
        tracker = KalmanBoxTracker(bbox)
        
        assert tracker.id >= 0
        assert tracker.hits == 0
        assert tracker.hit_streak == 0
        assert tracker.age == 0
        assert tracker.time_since_update == 0
    
    def test_tracker_update(self):
        """Test tracker update"""
        bbox = np.array([10, 10, 20, 20], dtype=np.float64)
        tracker = KalmanBoxTracker(bbox)
        
        initial_hits = tracker.hits
        tracker.update(bbox)
        
        assert tracker.hits == initial_hits + 1
        assert tracker.hit_streak > 0
        assert tracker.time_since_update == 0
    
    def test_tracker_predict(self):
        """Test tracker prediction"""
        bbox = np.array([10, 10, 20, 20], dtype=np.float64)
        tracker = KalmanBoxTracker(bbox)
        
        initial_age = tracker.age
        prediction = tracker.predict()
        
        assert tracker.age == initial_age + 1
        assert prediction.shape == (1, 4)
        assert tracker.time_since_update == 1


class TestSort:
    """Test SORT tracker"""
    
    def test_sort_initialization(self):
        """Test SORT initialization"""
        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        
        assert tracker.max_age == 5
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0
    
    def test_sort_update_empty(self):
        """Test SORT update with empty detections"""
        tracker = Sort()
        detections = np.empty((0, 5), dtype=np.float64)
        
        result = tracker.update(detections)
        
        assert result.shape == (0, 5)
        assert tracker.frame_count == 1
    
    def test_sort_update_with_detections(self):
        """Test SORT update with detections"""
        tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)
        detections = np.array([[10, 10, 20, 20, 0.8]], dtype=np.float64)
        
        result = tracker.update(detections)
        
        assert result.shape == (1, 5)
        assert tracker.frame_count == 1
        assert len(tracker.trackers) == 1


class TestCropBbox:
    """Test bounding box cropping"""
    
    def test_crop_bbox(self):
        """Test cropping image with bounding box"""
        # Create a test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:40, 10:30] = [255, 255, 255]  # White rectangle
        
        bbox = np.array([10, 20, 30, 40], dtype=np.float64)
        cropped = crop_bbox(image, bbox)
        
        assert cropped.shape == (20, 20, 3)  # Height: 40-20=20, Width: 30-10=20
        assert np.all(cropped[0, 0] == [255, 255, 255])  # Should contain white pixel


if __name__ == "__main__":
    pytest.main([__file__]) 