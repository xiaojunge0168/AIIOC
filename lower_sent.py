import sys
#from pipeline.NER.ner import label

def process(desc):
    desc = desc.lower()
    return desc

def process_file(f):
    output_f = f + ".lower"
    output_f = open(output_f, "w")
    with open(f, 'r') as f:
        for line in f:
            line = process(line)
            output_f.write(line)
    output_f.close()
    return output_f



if __name__ == "__main__":
    process_file(sys.argv[1])

