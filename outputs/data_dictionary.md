# Data Dictionary — Student Performance Prediction System (SPPS)

| Column | Type | Description |
|:---|:---|:---|
| student_ability | Float | Historical mean success rate of the student. |
| problem_difficulty | Float | Baseline success rate across all students for this step/problem. |
| Problem Hierarchy | Integer | Ordinal-encoded problem category hierarchy. |
| Step Duration (sec) | Float | Time taken by the student to complete the step. |
| engagement_ratio | Float | Total count of interactions per student (engagement proxy). |
| consistency_index | Float | Inverse of standard deviation of response times (higher = more consistent). |
| mastery_trend | Float | Expanding cumulative mean of Correct First Attempt (learning velocity). |
| timeliness | Binary (0/1) | 1 if response time is faster than median; 0 otherwise. |
| target | Binary (0/1) | Target variable: 1 = Correct First Attempt, 0 = Incorrect. |
