
import io

# parameters
data_path = "./datasets/harry-potter-1-2-4.txt"

with io.open(data_path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


print ('"'+ '","'.join(chars))
