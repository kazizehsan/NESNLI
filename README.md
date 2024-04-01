# Natural Explanation of SNLI

## Training Data
```
{'loss': 0.0915, 'learning_rate': 4.536929170549861e-05, 'epoch': 0.09}
{'eval_loss': 0.10310374200344086, 'eval_bleu': 0.817208404053622, 'eval_precisions': [0.8912066175148354, 0.8348841452289728, 0.8122569894463988, 0.7987220447284346], 'eval_brevity_penalty': 0.9804137377854408, 'eval_length_ratio': 0.9806030682419327, 'eval_translation_length': 5561, 'eval_reference_length': 5671, 'eval_runtime': 40.4007, 'eval_samples_per_second': 1.98, 'eval_steps_per_second': 0.495, 'epoch': 0.09}
{'loss': 0.0877, 'learning_rate': 4.382572227399814e-05, 'epoch': 0.12}
{'eval_loss': 0.10087231546640396, 'eval_bleu': 0.8166173724568622, 'eval_precisions': [0.8937477412359957, 0.8384671800513385, 0.8159657610718273, 0.8029845107669059], 'eval_brevity_penalty': 0.9755478645026058, 'eval_length_ratio': 0.9758420031740433, 'eval_translation_length': 5534, 'eval_reference_length': 5671, 'eval_runtime': 33.497, 'eval_samples_per_second': 2.388, 'eval_steps_per_second': 0.597, 'epoch': 0.12}
{'loss': 0.0878, 'learning_rate': 4.228215284249768e-05, 'epoch': 0.15}
{'eval_loss': 0.09932901710271835, 'eval_bleu': 0.8179369665023578, 'eval_precisions': [0.8942169540229885, 0.8352769679300291, 0.810465976331361, 0.7961711711711712], 'eval_brevity_penalty': 0.9816714850383143, 'eval_length_ratio': 0.9818374184447187, 'eval_translation_length': 5568, 'eval_reference_length': 5671, 'eval_runtime': 32.855, 'eval_samples_per_second': 2.435, 'eval_steps_per_second': 0.609, 'epoch': 0.15}
{'loss': 0.0846, 'learning_rate': 4.073858341099721e-05, 'epoch': 0.19}
{'eval_loss': 0.09888070821762085, 'eval_bleu': 0.8215943982897493, 'eval_precisions': [0.8973575408952005, 0.8398686850264454, 0.8151027207107163, 0.8016156302836747], 'eval_brevity_penalty': 0.9807732525171848, 'eval_length_ratio': 0.980955739728443, 'eval_translation_length': 5563, 'eval_reference_length': 5671, 'eval_runtime': 32.784, 'eval_samples_per_second': 2.44, 'eval_steps_per_second': 0.61, 'epoch': 0.19}
{'loss': 0.0837, 'learning_rate': 3.919501397949674e-05, 'epoch': 0.22}
{'eval_loss': 0.09699465334415436, 'eval_bleu': 0.8207668863506903, 'eval_precisions': [0.8977641543454742, 0.8432125869008416, 0.8167471221685852, 0.8032416132679985], 'eval_brevity_penalty': 0.9777133337599878, 'eval_length_ratio': 0.9779580320931053, 'eval_translation_length': 5546, 'eval_reference_length': 5671, 'eval_runtime': 32.5407, 'eval_samples_per_second': 2.458, 'eval_steps_per_second': 0.615, 'epoch': 0.22}
{'loss': 0.0825, 'learning_rate': 3.765144454799627e-05, 'epoch': 0.25}
{'eval_loss': 0.09605804830789566, 'eval_bleu': 0.8229811603009451, 'eval_precisions': [0.8992831541218638, 0.8398181818181818, 0.81309963099631, 0.797378277153558], 'eval_brevity_penalty': 0.9838240157299492, 'eval_length_ratio': 0.9839534473637807, 'eval_translation_length': 5580, 'eval_reference_length': 5671, 'eval_runtime': 33.2009, 'eval_samples_per_second': 2.41, 'eval_steps_per_second': 0.602, 'epoch': 0.25}
{'loss': 0.081, 'learning_rate': 3.610787511649581e-05, 'epoch': 0.28}
{'eval_loss': 0.09521929919719696, 'eval_bleu': 0.8201735830120996, 'eval_precisions': [0.899044528574004, 0.8412291933418694, 0.8160386114720624, 0.8017712455247786], 'eval_brevity_penalty': 0.9778935828927623, 'eval_length_ratio': 0.9781343678363604, 'eval_translation_length': 5547, 'eval_reference_length': 5671, 'eval_runtime': 33.049, 'eval_samples_per_second': 2.421, 'eval_steps_per_second': 0.605, 'epoch': 0.28}
{'loss': 0.0811, 'learning_rate': 3.456430568499534e-05, 'epoch': 0.31}
{'eval_loss': 0.09491034597158432, 'eval_bleu': 0.8223454228666467, 'eval_precisions': [0.9014414414414414, 0.843510054844607, 0.8181818181818182, 0.8020715630885122], 'eval_brevity_penalty': 0.9784341397183453, 'eval_length_ratio': 0.9786633750661259, 'eval_translation_length': 5550, 'eval_reference_length': 5671, 'eval_runtime': 32.7942, 'eval_samples_per_second': 2.439, 'eval_steps_per_second': 0.61, 'epoch': 0.31}
{'loss': 0.0804, 'learning_rate': 3.3020736253494876e-05, 'epoch': 0.34}
{'eval_loss': 0.09456899762153625, 'eval_bleu': 0.8225854526568122, 'eval_precisions': [0.9015151515151515, 0.8466325036603221, 0.8181649331352154, 0.8035444947209653], 'eval_brevity_penalty': 0.9773527401760503, 'eval_length_ratio': 0.977605360606595, 'eval_translation_length': 5544, 'eval_reference_length': 5671, 'eval_runtime': 32.8308, 'eval_samples_per_second': 2.437, 'eval_steps_per_second': 0.609, 'epoch': 0.34}
```
