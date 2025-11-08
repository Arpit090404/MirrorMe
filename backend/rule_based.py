def rule_based_scores(features):
    scores = {}
    
    # Blink rate scoring (more lenient - natural is 10-20 blinks/min)
    blink_rate = features.get('blink_mean', 0) * 60
    if 10 <= blink_rate <= 20: 
        scores['blink'] = 90  # Natural blinking
    elif 8 <= blink_rate < 10 or 20 < blink_rate <= 25:
        scores['blink'] = 75  # Slightly low/high but acceptable
    elif 5 <= blink_rate < 8 or 25 < blink_rate <= 30:
        scores['blink'] = 60  # Moderate issue
    elif blink_rate < 5:
        scores['blink'] = 70  # Too few blinks (not necessarily bad)
    else: 
        scores['blink'] = 45  # Excessive blinking (nervous)
    
    # Head stability scoring (more lenient - allow some natural movement)
    head_vel = abs(features.get('head_yaw_slope', 0)) + abs(features.get('head_pitch_slope', 0))
    if head_vel < 3: 
        scores['head_stability'] = 95  # Very stable
    elif head_vel < 6: 
        scores['head_stability'] = 85  # Good stability
    elif head_vel < 10: 
        scores['head_stability'] = 75  # Acceptable movement
    elif head_vel < 15: 
        scores['head_stability'] = 60  # Moderate movement
    else: 
        scores['head_stability'] = max(40, 90 - head_vel * 1.5)  # Excessive movement
    
    # Enhanced Gesture activity scoring using hand gesture features
    # Combine multiple hand gesture indicators
    hand_area = features.get('hand_area', 0)
    hand_motion = features.get('hand_motion', 0)
    finger_count = features.get('finger_count', 0)
    convexity_defects = features.get('convexity_defects', 0)
    hand_circularity = features.get('hand_circularity', 0)
    sift_keypoints = features.get('sift_keypoints', 0)
    
    # Normalize and combine gesture indicators
    # Hand motion (0-50 scale, normalized from 0-5 range)
    motion_score = min(50, (hand_motion / 5.0) * 50) if hand_motion > 0 else 0
    
    # Finger/gesture complexity (0-30 scale)
    complexity_score = min(30, (finger_count + convexity_defects) * 6) if (finger_count + convexity_defects) > 0 else 0
    
    # Hand presence and activity (0-20 scale)
    presence_score = min(20, (hand_area / 50000.0) * 20) if hand_area > 0 else 0
    
    # Total gesture activity (0-100)
    gesture_score = motion_score + complexity_score + presence_score
    
    # Use SIFT as backup if hand features are weak
    if gesture_score < 30 and sift_keypoints > 10:
        gesture_score = min(100, sift_keypoints / 2)
    
    # Boost gesture score if we have any hand activity at all
    if gesture_score > 0:
        gesture_score = min(100, gesture_score * 1.2)  # 20% boost for any activity
    
    scores['gesture_activity'] = max(40, min(100, gesture_score))  # Minimum 40 instead of 30
    
    # Gaze steadiness scoring (removed duplicate - use eye_contact below)
    # gaze_score = min(100, features.get('harris_corners', 0))
    # scores['gaze_steadiness'] = max(40, gaze_score)
    
    # Enhanced Eye contact scoring (based on gaze_offset and iris detection)
    gaze_offset = abs(features.get('gaze_offset', 0))
    iris_circles = features.get('iris_circles', 0)
    
    # Base score from gaze alignment (more lenient - allow some natural eye movement)
    if gaze_offset < 20:
        base_score = 95  # Excellent eye contact
    elif gaze_offset < 40:
        base_score = 85  # Good eye contact
    elif gaze_offset < 60:
        base_score = 75  # Acceptable
    elif gaze_offset < 90:
        base_score = 65  # Moderate
    else:
        base_score = 50  # Poor eye contact
    
    # Bonus for iris detection (indicates eye visibility)
    iris_bonus = min(10, iris_circles * 5) if iris_circles > 0 else -5
    
    scores['eye_contact'] = max(30, min(100, base_score + iris_bonus))
    
    # Enhanced Voice scoring (volume + pitch consistency)
    volume = features.get('volume_mean', 0)
    volume_std = features.get('volume_std', 0)
    pitch_mean = features.get('pitch_mean', 0)
    pitch_std = features.get('pitch_std', 0)
    hesitation = features.get('hesitation', 0)
    
    # Volume score (0-60) - more lenient thresholds
    if volume > 0.08:
        volume_score = 60  # Good volume
    elif volume > 0.05:
        volume_score = 55  # Acceptable
    elif volume > 0.03:
        volume_score = 50  # Moderate
    elif volume > 0.015:
        volume_score = 40  # Low but present
    else:
        volume_score = 25  # Very quiet
    
    # Consistency score (0-30) - lower std = more consistent
    volume_consistency = max(0, 30 - (volume_std * 100))
    pitch_consistency = max(0, 10 - (pitch_std / 10.0))
    
    # Hesitation penalty (0-10 deduction)
    hesitation_penalty = min(10, hesitation * 50)
    
    total_voice_score = volume_score + volume_consistency + pitch_consistency - hesitation_penalty
    
    scores['voice'] = max(30, min(100, total_voice_score))
    
    return scores