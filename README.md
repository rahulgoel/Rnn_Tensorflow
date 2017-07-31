# Rnn_Tensorflow
Contains a few examples for running RNNs using dynamicRNN in tf, taken from places on the web

get_files. sh: Downloads the 5 review files and word embeddings
python theme_extraction.py
Does:
1) Extraction based on heurisitics (noun phrases, adverbs, noun etc)
2) Ranks those using LDA and extracts topics
3) Uses that information in an seq2seq model for (probably) better summarization


Output:

('example candidate words', [u'essenti', u'advantag', u'promot', u'everyday', u'year', u'new', u'dishwash', u'mine', u'hand', u'someth', u'dishwash', u'anyway. i', u'spreader', u'butter', u'dough', u'cinnamon', u'roll', u'spread', u'pizza', u'sauc', u'pizza', u'coupl', u'time', u'frost', u'cupcak', u'great', u'tool', u'buck'])


('example candidate noun phrases', ['long time frontline plus customer', 'years', 'last year', 'generic equivalent. this product keeps ticks', 'dog', 'walks', 'lake', 'next morning', 'lots of ticks', 'dog dead', 'night', 'ones', "day earlier. i 've", 'new home', 'dog', 'laying around habits', 'wall', 'bedroom', 'office', 'baseboards', 'hair', 'magic eraser', 'much. fast forward', 'winter', 'spring', 'time', 'fresh (', 'softish ) with lots', 'hair', 'baseboard paint', 'dog', 'side', 'next day', 'dog', 'frontline plus', 'googling', 'other people', 'damage', "wood flooring. i 've", 'product', 'floors with wooden floors', 'baseboards', 'floors', 'virbac tick collar'])



Extraction Example 1 (Using words as inputs):
Topics:
[(0, u'0. 016*test + 0. 013*wire + 0. 012*kit + 0. 007*cabl + 0. 007*switch + 0. 007*oem + 0. 006*netgear + 0. 006*set + 0. 006*connect + 0. 006*oil'), (1, u'0. 053*hair + 0. 029*color + 0. 023*product + 0. 016*fish + 0. 013*great + 0. 010*spatula + 0. 010*tank + 0. 010*good + 0. 009*condition + 0. 008*ammonia'), (2, u'0. 095*cup + 0. 016*stainless + 0. 012*ring + 0. 012*whisk + 0. 011*bowl + 0. 011*easi + 0. 010*spoon + 0. 009*dishwash + 0. 009*time + 0. 008*size'), (3, u'0. 028*filter + 0. 019*good + 0. 016*qualiti + 0. 015*great + 0. 015*price + 0. 012*ear + 0. 011*year + 0. 010*headphon + 0. 010*sound + 0. 009*product'), (4, u'0. 030*tank + 0. 023*water + 0. 013*aquarium + 0. 008*tire + 0. 008*side + 0. 007*set + 0. 007*wheel + 0. 007*easi + 0. 007*cap + 0. 006*work'), (5, u'0. 018*food + 0. 013*fish + 0. 012*product + 0. 011*plant + 0. 011*water + 0. 010*label + 0. 009*cat + 0. 008*year + 0. 008*day + 0. 008*time'), (6, u'0. 029*handl + 0. 023*oxo + 0. 019*good + 0. 018*measur + 0. 015*set + 0. 014*easi + 0. 014*grip + 0. 014*great + 0. 010*product + 0. 010*nice'), (7, u'0. 016*dog + 0. 015*time + 0. 013*great + 0. 010*good + 0. 009*year + 0. 009*tender + 0. 008*thing + 0. 008*tape + 0. 008*easi + 0. 007*product'), (8, u'0. 035*product + 0. 025*skin + 0. 016*oil + 0. 013*nail + 0. 012*great + 0. 011*polish + 0. 011*good + 0. 010*time + 0. 010*dri + 0. 010*face'), (9, u'0. 023*headphon + 0. 022*batteri + 0. 014*radio + 0. 014*sound + 0. 012*soni + 0. 009*camera + 0. 009*great + 0. 008*music + 0. 008*bass + 0. 008*good')]

Top Words:

[u'test', u'wire', u'kit', u'cabl', u'switch']
[u'hair', u'color', u'product', u'fish', u'great']
[u'cup', u'stainless', u'ring', u'whisk', u'bowl']
[u'filter', u'good', u'qualiti', u'great', u'price']
[u'tank', u'water', u'aquarium', u'tire', u'side']
[u'food', u'fish', u'product', u'plant', u'water']
[u'handl', u'oxo', u'good', u'measur', u'set']
[u'dog', u'time', u'great', u'good', u'year']
[u'product', u'skin', u'oil', u'nail', u'great']
[u'headphon', u'batteri', u'radio', u'sound', u'soni']

Extraction Example 2 (Using noun phrases ans inputs): 

[(0, u'0. 031*hair + 0. 010*skin + 0. 010*day + 0. 007*dog + 0. 006*scent + 0. 005*stuff + 0. 005*months + 0. 005*face + 0. 004*dogs + 0. 004*week'), (1, u'0. 019*tank + 0. 016*handles + 0. 015*handle + 0. 008*price + 0. 008*time + 0. 006*oxo + 0. 006*spoons + 0. 006*years + 0. 005*job + 0. 005*set'), (2, u'0. 042*cups + 0. 017*cup + 0. 007*kitchen + 0. 007*dishwasher + 0. 006*oxo + 0. 006*stainless steel + 0. 006*filters + 0. 006*aquarium + 0. 005*years + 0. 005*hand'), (3, u'0. 012*water + 0. 012*car + 0. 012*filter + 0. 010*oil + 0. 010*product + 0. 006*time + 0. 005*nails + 0. 004*miles + 0. 004*measurements + 0. 004*glass'), (4, u'0. 029*headphones + 0. 009*bass + 0. 008*brush + 0. 006*years + 0. 006*pair + 0. 005*tanks + 0. 005*price + 0. 005*sound + 0. 004*filter + 0. 004*car'), (5, u'0. 051*product + 0. 014*skin + 0. 012*color + 0. 008*face + 0. 008*hands + 0. 007*years + 0. 006*price + 0. 006*products + 0. 006*bottle + 0. 005*time'), (6, u'0. 012*ring + 0. 007*food + 0. 007*time + 0. 005*unit + 0. 005*cups + 0. 005*ears + 0. 005*way + 0. 005*works + 0. 004*years + 0. 004*pump'), (7, u'0. 010*radio + 0. 010*quot + 0. 010*sound + 0. 007*batteries + 0. 005*headphones + 0. 005*volume + 0. 004*sony + 0. 004*spoon + 0. 004*koss + 0. 004*case'), (8, u'0. 012*battery + 0. 008*camera + 0. 007*keyboard + 0. 005*drawer + 0. 005*charger + 0. 004*computer + 0. 004*price + 0. 004*cans + 0. 004*highs + 0. 003*sony'), (9, u'0. 010*switch + 0. 007*set + 0. 006*problem + 0. 004*price + 0. 004*time + 0. 004*phones + 0. 003*unit + 0. 003*way + 0. 003*router + 0. 003*numbers')]
[u'hair', u'skin', u'day', u'dog', u'scent']
[u'tank', u'handles', u'handle', u'price', u'time']
[u'cups', u'cup', u'kitchen', u'dishwasher', u'oxo']
[u'water', u'car', u'filter', u'oil', u'product']
[u'headphones', u'bass', u'brush', u'years', u'pair']
[u'product', u'skin', u'color', u'face', u'hands']
[u'ring', u'food', u'time', u'unit', u'cups']
[u'radio', u'quot', u'sound', u'batteries', u'headphones']
[u'battery', u'camera', u'keyboard', u'drawer', u'charger']
[u'switch', u'set', u'problem', u'price', u'time']5B

# 
