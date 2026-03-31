# Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style

By: [Connor Baumler](https://ctbaumler.github.io/) `<baumler@cs.umd.edu>`, Calvin Bao, Huy Nghiem, Xinchen Yang, Marine Carpuat, and Hal Daumé III

```
Bibtex will be updated here when available.
```

This repository contains writing logs generated in our study about post-editing LLM-generate text to express personal style. 

Each log contains the following:

- `user_info` consisting of:
  - A unique user `id` (same as the file name)
  - `start_time` when they began the study
  - A `conditions` list of length 6 with 1's representing treatment tasks (post-editing) and 0's representing control tasks (unassisted human writing)
  - A `scenarios` list of length 6 showing the writing scenarios each participant completed in order.
- `responses` consisting of a dictionary of the logs for each of the 6 writing tasks completed. Each value contains:
  - A `start_time` when they began this task
  - The `scenario` written about for this task
  - A `model_generation_shown` flag that is 1 if the user was provided a LLM-generated draft to post-edit (i.e., if they were in the treatment block)
  - A string `details` containing the list of content details to be provided to the LLM separated by new line characters
  - A string `model_generation` containing the LLM-generated draft. Note that this field will be populated even if the user never saw this draft (i.e., if they were in the control block)
  - A `submit_details_time` when they submitted the content details and moved on to writing or post-editing
  - A `final_version` string containing the final human-written or human post-edited document.
  - A list of `edits` made during writing or post-editing. Each action is categorized with a `type` in `{'insert', 'delete', 'replace'}` and contains the character-level `position` of the edit, the text `removed` or `added` and a `timestamp`. 
- `pre-survey` data consists of answer to three 5-point likert scale questions:
  - `conf`: "How confident are you in your ability to edit AI-written text to match your own voice?" where 1 is `Very Unconfident` and 5 is `Very Confident`
  - `likely_i`: How likely are you to use AIs for writing tasks where capturing your voice is important to you? where 1 is `Very Unlikely` and 5 is `Very Likely`
  - `likely_ni`: How likely are you to use AIs for writing tasks where capturing your voice is *not* important to you? where 1 is `Very Unlikely` and 5 is `Very Likely`
- `mid_survey_1` and `mid_survey_2` contain identical questions asked after each block. They contain a `submit_time`, a `condition` flag (1 if this mid-survey is about the treatment block and 0 if it is about the control block), and five 20-point TLX-style questions:
  - `mental`: "How mentally demanding were the writing tasks?" from `Not Demanding at all` to `Very Demanding`
  - `hard`: "How hard did you have to work to write at your level of performance?" from `Not Hard at all` to `Very Hard`
  - `insecure`: "How insecure, discouraged, irritated, stressed, and annoyed were you during the task?" from `Not insecure/etc at all` to `Very insecure/etc`
  - `performance`: "How successful were you in the writing tasks?" from `Very Successful` to `Very Unsuccessful`
  - `temporal`: "How long did it feel like the task was taking?" from `Very Short` to `Very Long`
- `post_survey` data repeats the three likert-scale questions from the pre-survey, and contains a `submit_time`, a free-text `feedback` section, and responses to two other sets of questions.
  - Questions about future preference to post-edit:
    - `future_pref` contains either `postedit`, `alone`, or `model` based on whether the user would rather post-edit LLM drafts, write alone, or use LLM drafts without post-editing in future writing tasks where personal style is important. (In the study, participants were also given a write-in option, but this is removed here since it was never chosen)
    - `rankings` contains how much 7 factors influenced their preference on how to write in the future. These options consisted of `["Originality","Style","Overall_quality","Efficiency","Reliability","Privacy","Ownership"]` and are listed from most influential to least influential. If participants said they would prefer to post-edit, the items were framed positively (e.g., "The original AI drafts were original or interesting. They sounded better than what I would have come up with on my own."), and if not, they were framed negatively (e.g., "The original AI drafts were too cliche or unoriginal. They weren't any more interesting or original than what I would have come up with alone.")
  - Demographic questions were all optional. Race and gender questions allowed for multi-select and gender for write-ins but these features were not used in practice.
    - We collected gender (`gender`: woman/man/non-binary and `trans`: trans/cis), race/ethnicity (`race`: American Indian or Alaska Native/Asian/Black or African American/Native Hawaiian or Other Pacific Islander/White and `hisp`: yes/no about whether they are Hispanic or Latino ), level of education (`loe`: Didn’t Finish High School/Didn’t Finish High School/but completed a technical/vocational program/High School Graduate or GED (General Education Diploma)/Completed High School and a technical/vocational program/Less than 2 years of college/2 Years of College or more/ including associate degree or equivalent/College graduate (4 or 5 year program)/Master’s degree (or other post-graduate training)/Doctoral degree (PhD, MD, JD, etc.)), and `age` (integer)
    - For each of `gender`, `race`, `loe`, and `age`, we list a likert `{x}_table` score representing for the given characteristic how important it is to the user that this part of their identity does or does not come across in the writing tasks completed in this study where 1 is "Very important that it does not come across", 3 is "It doesn't matter to me how readers perceive this characteristic", and 5 is "Very important that it does come across"
    - `present_other` and `mask_other` contain freetext responses about any other characteristics the user wants to be inferrable/present in their writing or masked.