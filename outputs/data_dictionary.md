# Data Dictionary - Student Performance Prediction
| Column | Type | Description |
| :--- | :--- | :--- |
| Anon Student Id | String | Unique student identifier. |
| student_ability | Float | Historical success rate of the student (mean). |
| problem_difficulty | Float | Baseline success rate across all students for this step. |
| engagement_ratio | Float | Count of interactions as a proxy for engagement. |
| consistency_index | Float | Inverse of standard deviation of response times. |
| mastery_trend | Float | Expanding mean of Correct First Attempt (learning velocity). |
| timeliness | Binary | 1 if response is faster than median; otherwise 0. |
| Correct First Attempt | Integer | Target: 1 for correct, 0 for incorrect. |
| latitude/longitude | Float | Synthetic coordinates for spatial risk analysis. |
| geo_region | String | School district assignment for the student. |
