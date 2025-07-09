import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import cv2
from typing import List, Tuple, Optional, Union, Any
import numpy.typing as npt

def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n random colors for visualization"""
    colors = []
    for i in range(n):
        colors.append((np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))
    return colors

def convert_bbox_to_z(bbox: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert bounding box [x1,y1,x2,y2] to state vector [u,v,s,r]
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        State vector [u, v, s, r] where:
        u, v: center coordinates
        s: area (width * height)
        r: aspect ratio (width / height)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x: npt.NDArray[np.float64], score: Optional[float] = None) -> npt.NDArray[np.float64]:
    """
    Convert state vector [u,v,s,r] to bounding box [x1,y1,x2,y2]
    
    Args:
        x: State vector [u, v, s, r]
        score: Optional confidence score
        
    Returns:
        Bounding box coordinates [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))

class KalmanBoxTracker:
    """
    Kalman Filter tracker for bounding box tracking
    """
    count: int = 0
    
    def __init__(self, bbox: npt.NDArray[np.float64]) -> None:
        """
        Initialize Kalman Filter tracker
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],  [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update: int = 0
        self.id: int = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[npt.NDArray[np.float64]] = []
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0

    def update(self, bbox: npt.NDArray[np.float64]) -> None:
        """
        Update tracker with new bounding box observation
        
        Args:
            bbox: New bounding box [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> npt.NDArray[np.float64]:
        """
        Predict next state using Kalman Filter
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> npt.NDArray[np.float64]:
        """
        Get current state estimate
        
        Returns:
            Current bounding box estimate [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)

def iou(bb_test: npt.NDArray[np.float64], bb_gt: npt.NDArray[np.float64]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bb_test: First bounding box [x1, y1, x2, y2]
        bb_gt: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

def associate_detections_to_trackers(
    detections: npt.NDArray[np.float64], 
    trackers: npt.NDArray[np.float64], 
    iou_threshold: float = 0.3
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Associate detections with trackers based on IoU using Hungarian algorithm
    
    Args:
        detections: Array of detections [x1, y1, x2, y2, confidence]
        trackers: Array of tracker predictions [x1, y1, x2, y2, confidence]
        iou_threshold: Minimum IoU threshold for association
        
    Returns:
        Tuple of (matches, unmatched_detections, unmatched_trackers)
    """
    if len(trackers) == 0:
        return np.array([]), np.arange(len(detections)), np.array([])

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind)))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort:
    """
    Deep SORT tracker implementation
    """
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        """
        Initialize SORT tracker
        
        Args:
            max_age: Maximum number of frames to keep track without detection
            min_hits: Minimum number of hits to confirm track
            iou_threshold: IoU threshold for association
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def update(self, dets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Update tracker with new detections
        
        Args:
            dets: Array of detections [x1, y1, x2, y2, confidence]
            
        Returns:
            Array of tracked objects [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        # Get predicted locations from trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Assign detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers
        for t, trk in enumerate(self.trackers):
            if t in unmatched_trks:
                continue
            d = matched[np.where(matched[:, 1] == t)[0], 0]
            trk.update(dets[d, :][0])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

def draw_tracks(
    frame: npt.NDArray[np.uint8], 
    tracks: npt.NDArray[np.float64], 
    class_names: List[str], 
    colors: List[Tuple[int, int, int]]
) -> npt.NDArray[np.uint8]:
    """
    Draw tracked objects on frame with bounding boxes and labels
    
    Args:
        frame: Input image frame
        tracks: Array of tracked objects [x1, y1, x2, y2, track_id]
        class_names: List of class names
        colors: List of colors for visualization
        
    Returns:
        Frame with drawn tracks
    """
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        # Choose color based on track_id
        color = colors[track_id % len(colors)]
        
        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with track_id
        label = f"ID: {track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def crop_bbox(image: npt.NDArray[np.uint8], bbox: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """
    Crop image according to bounding box coordinates
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Cropped image region
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    return image[y1:y2, x1:x2] 