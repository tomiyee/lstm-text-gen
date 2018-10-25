
import io


# Takes Command line inputs to override the above
if __name__ == "__main__":
   argv = sys.argv[1:]

   try:
       opts, args = getopt.getopt(argv,"hd:c:e:l:n:",["dataset=","checkpoints=","epochs=","load_model=","name="])
   except getopt.GetoptError:
       print ('test.py -i <inputfile> -o <outputfile>')
       sys.exit(2)

   for opt, arg in opts:

       # Help Command
       if opt == '-h':
           print ('charset.py -d <pathtodataset> -s <pathtosave>')
           sys.exit()

       # Dataset Name
       elif opt in ("-d", "--dataset"):
           dataset_path = "./datasets/" + arg

# parameters
data_path = "./datasets/harry-potter-1-2-4.txt"

with io.open(data_path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

f = open("charset.txt","w+")
charset = '["'+ '","'.join(chars) + '"]'
f.write(charset)
f.close()

print (charset)
