from transformers import TFBertModel as _
from transformers import BertConfig

def main():
    config_base = BertConfig.from_pretrained("/home/v-jundapan/data/models/bert-base-uncased")
    print(config_base)
    
    config_large = BertConfig.from_pretrained("/home/v-jundapan/data/models/bert-large-uncased")
    print(config_large)
    
if __name__ == "__main__":
    main()