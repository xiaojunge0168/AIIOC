import sys
#from pipeline.NER.ner import label

def process(desc):
    words = desc.split()
    desc = ""
    for word in words:
        if word[0] == '(' and word[-1] == ')':
            word = word[1:-1]
        elif word[0] in ['"',"'", ';','(',')','[',']']:
            word = word[1:]
        elif word[-1] in ['"',"'", ';','(',')','[',']']:
            word = word[:-1]
        if not word: continue
        word = word.lower()
        if word[0] in [',','.']:
            desc += str(word[0]) + " " + str(word[1:]) + " "
        elif word[-2:] in ['),',').','".','",', ');']:
            desc += str(word[:-2]) + " " + str(word[-1]) + " "
        elif word[-1] in [',', '"', '.', ';', ')']:
            desc += str(word[:-1]) + " " + str(word[-1]) + " "
        else:
            desc += word + " "
    return desc

def lower_process(desc):
    desc = desc.lower()
    return desc
# def process_v2(desc):
#     formalized = ""
#     for index,ch in desc:
#         if ch in [')','(','[',']','\\','/','"',';']:
#             continue
#         if ch in [',','.']:
#             if index>=len(desc) or desc[index+1]==' ':
#                 formalized += " " + str(ch)
#
def process_file(f):
    output_f = f + ".processed"
    output_f = open(output_f, "w")
    with open(f, 'r') as f:
        for line in f:
            line = process(line)
            output_f.write(line+'\n')
    output_f.close()
    return output_f



if __name__ == "__main__":
    process_file(sys.argv[1])

