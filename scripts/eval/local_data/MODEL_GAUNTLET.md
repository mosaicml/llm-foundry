# Mosaic Model Gauntlet v0 - Evaluation Suite


<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
<p align="center">
    <picture>
      <img alt="LLM Foundry" src="../../../assets/radar_blog.png" width="60%">
    </picture>
    <br>
    MPT-7B vs MPT-30B compared on the 6 categories of Model Gauntlet.
</p>
The Mosaic Model Gauntlet is MosaicML’s new technique for evaluating the quality of pretrained foundation models. The Model Gauntlet encompasses 35 different benchmarks collected from a variety of sources, and organized into 6 broad categories of competency that we expect good foundation models to have. We compiled the categories after an extensive review of existing LLM publications, and open source evaluation harnesses such as EleutherAI Eval Harness and Stanford CRFM’s HELM.

<br>
While deciding which benchmarks to include, we had a few criteria in mind. We wanted benchmarks to require a broad range of skills that were useful for practical applications, we wanted them to come from a diverse range of sources, we wanted them to capture skills that have been traditionally emphasized by the research community as well as those that have been underexplored, and we wanted them to be evaluated via simple, unambiguous metrics such as exact match and multiple choice accuracy. The philosophy behind compiling aggregate scores as opposed to the more common approach of reporting individual metrics, is two-fold.

1. LLMs are the first generation of true generalist models. They are sought after because they can do tens of thousands of different tasks well with only a little bit of prompting. This capacity cannot be adequately captured with only 10 different academic benchmarks. It wouldn't be possible to define and evaluate every conceivable task an LLM could be used for, but with Composer's blazing fast ICL evaluation, it is certainly worth a shot!

2. Using a small set of metrics can be produce misleadingly high variance. This can lead us into thinking models' abilities are starkly different, even when a few minutes of interacting with the models will reveal that they feel pretty similar. By aggregating scores across dozens of benchmarks, we can reduce the signal-to-noise ratio and get more robust estimates of models' relative performance.

Furthermore, we made the unique design decision to report multiple separate aggregate scores organized by category, rather than a single monolithic metric. This is because end users frequently want to deploy models for specific use-cases. There are many design decisions that go into training a model for a specific vertical, including dataset mixture. By decomposing model performance into a manageable set of core competencies, we can make more intelligent design decisions during the model training process.

At evaluation time, we run all the benchmarks, average the subscores within each category, and report the composite score for each category as well as the average of the composite scores. Some benchmarks are multiple choice examinations, for which it is possible to get above 0% accuracy with just random guessing. In order to ensure that all our composite scores are less than or equal to 1, we subtract the random baseline accuracy and rescale the remainder by 1 minus the baseline accuracy.

For example, if benchmark A has a random baseline accuracy of 25%, and the model achieved 30%, we would report this as (0.3 - 0.25)/(1-0.25) = 0.0667. This can be thought of as the accuracy above chance rescaled so the max is 1. For benchmarks in which the random guessing baseline accuracy is ~0 we report the accuracy as is. Note that with this rescaling, a model could technically score below 0 on a category as a whole, but we haven’t found this to occur with any of the models we’ve tested.

This is version v0, in the coming weeks we will update the mixture to include more benchmarks.

### Reading Comprehension

Reading comprehension benchmarks test a model’s ability to answer questions based on the information in a passage of text. It includes a number of multiple choice benchmarks as well as some benchmarks that require the model to output an exact correct answer.

1. BIG-bench: Understanding fables
   - Description: Understanding fables consists of 189 short stories followed by 4-choice multiple choice in which the model must select the correct moral for the story.
   - Year released: 2022
   - Number of few shot examples: 10
   - Random baseline accuracy: 25%

2. Pubmed QA Labeled
   - Description: Pubmed QA L consists of 1000 hand-labeled medical documents followed by a related question for which the model must respond yes/no/maybe.
   - Year released: 2019
   - Number of few shot examples: 10
   - Random baseline accuracy: ~0%

3. SQuAD
   - Description: SQuAD consists of 10,570 short documents followed by a related question. The documents range from short news clippings about sports events, to blurbs explaining concepts in physics, to documents about US history. We expect the model to output the exact correct answer.
   - Year released: 2016
   - Number of few shot examples: 10
   - Random baseline accuracy: ~0%


4. BoolQ
   - Description: BoolQ consists of 3,270 short passages on a diverse range of subjects followed by a yes/no questions. The model is expected to answer in multiple-choice format.
   - Year released: 2019
   - Number of few shot examples: 10
   - Random baseline accuracy: ~50%


### Commonsense Reasoning

Commonsense reasoning tests the models’ ability to perform basic reasoning tasks that require commonsense knowledge of objects, their properties, and their behavior.

5. BIG-bench: Strategy QA
   - Description: BIG-bench strategy QA consists of 2,289 very eclectic yes/no questions on a wide range of commonsense subjects, e.g “Can fish get Tonsilitis?”
   - Year released: 2022
   - Number of few shot examples: 10

6. BIG-bench: Strange Stories
   - Description: BIG-bench strange stories consists of 174 short stories followed by a two-choice multiply choice question in which the model is asked to make commonsense inferences about the characters in the stories, how they might feel, and why they act in certain ways.
   - Year released: 2022
   - Number of few shot examples: 10
   - Random baseline accuracy: 50%

7. BIG-bench: Novel Concepts
   - Description: BIG-bench novel concepts consists or 32 find-the-common-concept problems in which the model is given 3 words and has to choose from among 4 possible concepts that they all have in common.
   - Year released: 2022
   - Number of few shot examples: 10
   - Random baseline accuracy: 25%

8. COPA
   - Description: COPA consists of 100 cause/effect multiple choice questions in which the model is prompted with a premise and the model must choose correctly between two possible causes/effects of the premise.
   - Year released: 2011
   - Number of few shot examples: 0
   - Random baseline accuracy: 50%

9. PIQA
    - Description: PIQA consists of 1,838 commonsense physical intuition 2-choice multiple choice questions.
    - Year released: 2019
    - Number of few shot examples: 10
    - Random baseline accuracy: 50%

10. OpenBook QA
    - Description: OpenBook QA consists of 500 four-choice multiple choice questions that rely on basic physical and scientific intuition about common objects and entities.
    - Year released: 2018
    - Number of few shot examples: 0
    - Random baseline accuracy: 25%


### World knowledge

11. Jeopardy
    - Description: Jeopardy consists of 2,117 Jeopardy questions separated into 5 categories: Literature, American History, World History, Word Origins, and Science. The model is expected to give the exact correct response to the question. It was custom curated by MosaicML from a larger Jeopardy set available on [Huggingface](https://huggingface.co/datasets/jeopardy).
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

12. MMLU
    - Description: MMLU consists of 14,042 four-choice multiple choice questions distributed across 57 categories. The questions are in the style of academic standardized tests and the model is provided the question and the choices and is expected to choose between A, B, C, and D as its outputs. The subjects range from jurisprudence, to math, to morality.
    - Year released: 2019
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%


13. BIG-bench: wikidata
    - Description: BIG-bench wikidata consists of 20,321 questions regarding factual information pulled from wikipedia. Questions range from the native language of celebrities to the country that different regions belong to. Models are given a sentence such as “The country of citizenship of Barack Obama is” and are expected to complete the sentence with e.g. “the United States.”
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: ~0%

14. ARC easy
    - Description: ARC easy consists of 2,376 easy four-choice multiple choice science questions drawn from grade 3-9 science exams. The questions rely on world knowledge related to basic science.
    - Year released: 2019
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

15. ARC challenge
    - Description: ARC easy consists of 2,376 easy four-choice multiple choice science questions drawn from grade 3-9 science exams. The questions rely on scientific world knowledge and some procedural reasoning.
    - Year released: 2019
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

16. BIG-bench misconceptions
    - Description: Big bench misconceptions consists of 219 true or false questions regarding common misconceptions about a variety of topics including urban legends, stereotypes, basic science, and law.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 50%


### Symbolic problem solving

Symbolic problem solving tasks test the model’s ability to solve a diverse range of symbolic tasks including arithmetic, logical reasoning, algorithms, and algebra.

17. BIG-bench elementary math QA
    - Description: Big bench elementary math QA consists of 38,160 four-choice multiple choice arithmetic word problems.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

18. BIG-bench dyck languages
    - Description: Big bench dyck languages consists of 1000 complete-the-sequence questions, in which a partially completed balanced expression consisting of parentheses and braces is given, and the model needs to output the exact tokens necessary in order to complete the balanced expression.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

19. BIG-bench dyck languages
    - Description: Big bench dyck languages consists of 1000 complete-the-sequence questions, in which a partially completed balanced expression consisting of parentheses and braces is given, and the model needs to output the exact tokens necessary in order to complete the balanced expression.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

20. BIG-bench cs algorithms 
    - Description: Big bench cs algorithms consists of 1,320 samples of questions falling into one of two types. In the first type the model must determine the length of the longest common subsequence of two strings, and in the second type the model must determine whether an expression consisting of parentheses and braces is balanced.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

21. BIG-bench logical deduction
    - Description: Big bench logical deduction consists of 1500 four-choice multiple choice questions, in which the model is posed with a number of logical constraints describing the relative ordering of some number of objects. The model must then choose from among a list of four statements, which statement is the only one that is logically consistent with the constraints posed.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

22. BIG-bench operators
    - Description: Big bench logical operators consists of 210 questions, in which a number of mathematical operators are defined and the model is expected to calculate the result of some expression consisting of those defined operators. This tests the model’s ability to handle mathematical abstractions and apply them appropriately.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

23. BIG-bench repeat copy logic
    - Description: Big bench repeat copy logic consists of 32 tasks in which the model is commanded to repeat some combination of words some number of times in a particular order, and the model is expected to output the correct result.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

24. Simple arithmetic with spaces
    - Description: Simple arithmetic with spaces was developed by MosaicML. It consists of 1000 arithmetic problems consisting of up to 3 operations and using numbers of up to 3 digits. There is spacing between all numbers and operators. The model is expected to calculate the correct result of the expression using the appropriate order of operations.
    - Year released: 2023
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

25. Simple arithmetic without spaces
    - Description: Simple arithmetic with spaces was developed by MosaicML. It consists of 1000 arithmetic problems consisting of up to 3 operations and using numbers of up to 3 digits. There is no spacing between any of the numbers and operators. The model is expected to calculate the correct result of the expression using the appropriate order of operations.
    - Year released: 2023
    - Number of few shot examples: 10
    - Random baseline accuracy: 0%

26. Math QA
    - Description: Math QA consists of 2,983 four-choice multiple choice math word problems. The questions require basic reasoning, language comprehension, and arithmetic/algebraic skills.
    - Year released: 2021
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

27. LogiQA
    - Description: LogiQA consists of 651 four-choice multiple choice logical word problems.The questions involve making logical deductions based on mathematical and symbolic descriptions of problems.
    - Year released: 2020
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%


### Language understanding

Language understanding tasks evaluate the model’s ability to understand the structure and properties of languages, including determining which types of sentences are more likely than others, performing translations between languages, and identifying languages.

28. LAMBADA
    - Description: LAMBADA consists of 5,153 passages take from books. The model is expected to read the first N-1 words of each passage and predict the final token.
    - Year released: 2016
    - Number of few shot examples: 0
    - Random baseline accuracy: 0%

29. HellaSwag
    - Description: HellaSwag consists of 10,042 multiple choice scenarios in which the model is prompted with a scenario and choose the most likely conclusion to the scenario from four possible options.
    - Year released: 2019
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

30. Winograd Schema Challenge
    - Description: The Winograd Schema Challenge consists of 273 scenarios in which the model must use semantics to correctly resolve the anaphora in a sentence. Two possible beginnings to a sentence are presented as well as an ending. Both involve some anaphora being resolved in a different way, only one of which would be semantically valid, and the model must choose which option produces the valid resolution.
    - Year released: 2012
    - Number of few shot examples: 0
    - Random baseline accuracy: 50%

31. Winogrande
    - Description: The Winogrande consists of 1,267 scenarios in which two possible beginnings of a sentence are presented along with a single ending. Both combinations are syntactically valid, but only one is semantically valid, and the model must choose the one that is semantically valid.
    - Year released: 2012
    - Number of few shot examples: 0
    - Random baseline accuracy: 50%

32. BIG bench language identification
    - Description: BIG bench language identification consists of 10,000 four-choice multiple choice questions in which a sentence in some language besides english is presented and the model is prompted to identify the language of the sentence amongst four options.
    - Year released: 2012
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

33. BIG bench conceptual combinations
    - Description: BIG bench conceptual combinations consists of 103 four-choice multiple choice questions in which the model is presented with a made up word and its definition along with a multiple choice question regarding the meaning of a sentence using that made up word. The model is then expected to select the correct answer among the choices presented.
    - Year released: 2022
    - Number of few shot examples: 10
    - Random baseline accuracy: 25%

34. BIG bench conlang translation
    - Description: BIG bench conlang translation consists of 164 example problems in which the model is given translations of simple sentences between English and some fake constructed language. The model is then tested for its ability to translate a complex sentence in the fake language into English.
    - Year released: 2022
    - Number of few shot examples: 0
    - Random baseline accuracy: 0%


### Programming
Programming tasks evaluate the model's ability to understand code, write functionally correct code given a specification, simulate code, and document code. Right now we just have HumanEval but later versions will include more.

35. HumanEval code generation
    - Description: HumanEval consists of 164 python programming challenges, in which the model is presented with the method signature and docstring comment for a python program and is expected to complete the program. We then test the resultant code’s functional correctness on a number of test input/output pairs.
    - Year released: 2022
    - Number of few shot examples: 0
    - Random baseline accuracy: 0%
