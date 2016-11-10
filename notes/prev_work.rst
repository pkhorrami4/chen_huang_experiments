[1] - Audio-visual Affect Recognition through Multi-stream Fused HMM for HCI 
Zhihong Zeng, Jilin Tu, Brian Pianfetti, Ming Liu, Tong Zhang, Zhenqiu Zhang,  
Thomas S. Huang and Stephen Levinson

- Multi-modal (images+audio)
- Person independent (Leave-One-Out CV)
- 5 Methods
	1. Face-only HMM
	2. Pitch-only HMM
	3. Energy-only HMM
	4. Independent-HMM
	5. MFHMM
	
+----------+--------+---------+----------+--------+--------+
|	   |  Face  |  Pitch  |  Energy  |  IHMM  |  FHMM  |
+----------+--------+---------+----------+--------+--------+
|11 states |  38.64 |  57.27  |  66.36   |  72.42 |  80.61 |
+----------+--------+---------+----------+--------+--------+
| 7 states |  52.38 |  63.81  |  70.71   |  75.24 |  85.24 |
+----------+--------+---------+----------+--------+--------+

[2] - Multi-stream Confidence Analysis for Audio-visual Affect Recognition 
Zhihong Zeng, Jilin Tu, Ming Liu and Thomas S. Huang 

- Multi-model (images+audio)
- Person independent (Leave-One-Out CV)
	- 14 subj. for train, 5 for valid., 1 for test.
- 6 Methods
	1. Face-only HMM
	2. Pitch-only HMM
	3. Energy-only HMM
	4. MFHMM1
	5. MFHMM2
	6. MFHMM3

+----------+--------+---------+----------+---------+---------+---------+
|          |  Face  |  Pitch  |  Energy  |  MHMM1  |  MHMM2  |  MHMM3  |
+----------+--------+---------+----------+---------+---------+---------+
|11 states |  0.39  |  0.60   |   0.69   |   0.70  |   0.72  |   0.75  | 
+----------+--------+---------+----------+---------+---------+---------+

[3] - FACIAL EXPRESSION RECOGNITION FROM VIDEO SEQUENCES
Ira Cohen, Nicu Sebe, Ashutosh Garg, Michael S. Lew, Thomas S. Huang

- Tree Augmented Naive Bayes (TAN) classifier
- 6 basic emotions
- Person Dependent + Person Independent


[4] - Facial expression recognition from video sequences: 
temporal and static modeling
Ira Cohen, Nicu Sebe, Ashutosh Garg, Lawrence S. Chen, Thomas S. Huang

- 5 subjects
- 6 basic emotions
- Person Dependent + Person Independent


[5] - Emotion Recognition from Facial Expressions using Multilevel HMM
Ira Cohen, Ashutosh Garg, Thomas S. Huang

- Video-only features
- Person Dependent + Person Independent
- 5 subjects
- 6 basic emotions

Person Dependent Experiments (5 subj, 6 emotions)

+------+------------+----------------+
| Subj | Single HMM | Multilevel HMM |
+------+------------+----------------+
|  1   |    82.86%  |       80%      |
+------+------------+----------------+
|  2   |    91.43%  |      85.71%    |
+------+------------+----------------+
|  3   |    80.56%  |      80.56%    |
+------+------------+----------------+
|  4   |    83.33%  |      88.89%    |
+------+------------+----------------+
|  5   |    54.29%  |      77.14%    |
+------+------------+----------------+
| Total|    78.49%  |      82.46%    |
+------+------------+----------------+

Person Independent

+------------+----------------+
| Single HMM | Multilevel HMM |
+------------+----------------+
|    55%     |       58%      |
+------------+----------------+

[6] - Emotion Recognition Based on Joint Visual and Audio Cues 
Nicu Sebe, Ira Cohen, Theo Gevers, Thomas S. Huang

- Uses 38 subjects (?)
