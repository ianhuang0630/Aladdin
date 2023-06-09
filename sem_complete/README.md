# Semantic Autocompletion

We use GPT-3 to infer the set of objects that exists within a scene, given a relatively abstract scene description like: `an abandoned battlefield in Ukraine`.

This module queries GPT-3 for more fine-grained stylistic/material descriptors of shapes, used for downstream shape retrieval and texture editing.


First, put your openAI API key (https://beta.openai.com/account/api-keys) inside `credentials/openai_key`. E.g. `mkdir credentials && echo "sk-tdaVyEtHiSiSNotAReALopEnAIaPiKey0sadf012dFfdsVVL" > credentials/openai_key`

Run the following as an example:
```
python scripts/simple_demo.py iterative "an abandoned living room"
```
It should output a hierarchical enumeration of the visual attributes of assets that one would expect to find within this space. 




