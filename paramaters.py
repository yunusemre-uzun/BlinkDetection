from imutils import face_utils

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold

EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
LIVENESS_THRESH = 10

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]