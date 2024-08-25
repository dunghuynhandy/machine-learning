from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from data import TranslationDataLoader
from loguru import logger
from model import Transformer
import torch
from torch import optim, nn
from torch.optim import Adam
from tqdm import tqdm
logger.remove()

def evaluate(model, valid_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    
    with torch.no_grad():  # Disable gradient computation
        for i, valid_batch in enumerate(valid_loader):
            src, trg = valid_batch
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:, :-1])
            output = output.view(-1, output.size(-1))
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            val_loss += loss.item()
    
    val_loss /= i  # Average the validation loss
    return val_loss

def translate_sentence(model, sentence, data_loader, device, max_len=50):
    model.eval()
    
    # Tokenize the source sentence
    tokens = data_loader.token_transform[data_loader.src_language](sentence)
    tokens = torch.tensor([data_loader.vocab_transform[data_loader.src_language](tokens)], dtype=torch.long).to(device)
    
    # Add <bos> and <eos> tokens
    tokens = tokens.to("cpu")
    src_tensor = data_loader.tensor_transform(tokens[0]).unsqueeze(0).to(device)  # Add batch dimension
    # Start with the <bos> token
    trg_tensor = torch.tensor([[data_loader.BOS_IDX]], dtype=torch.long).to(device)
    
    # Iteratively decode the output
    
    for _ in range(max_len):
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        
        # Get the token with the highest probability
        next_token = output.argmax(2)[:, -1].item()
        
        # Append the token to the target sequence
        trg_tensor = torch.cat([trg_tensor, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        
        # Break if <eos> token is generated
        if next_token == data_loader.EOS_IDX:
            break
    
    # Convert tokens back to text
    translated_tokens = trg_tensor.squeeze().tolist()[1:-1]  # Exclude <bos> and <eos>
    translated_text = ' '.join([data_loader.vocab_transform[data_loader.trg_language].lookup_token(token) for token in translated_tokens])
    
    return translated_text
def inference_examples(model, data_loader, device):
        # examples = [
        #     "Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.",
        #     "Zwei Kinder spielen im Park.",
        #     "Eine Gruppe von Menschen geht auf einem Bürgersteig."
        # ]
        
        examples = [
            "A man in a blue shirt is standing on a ladder and cleaning a window.",
            "Two children are playing in the park.",
            "A group of people is walking on a sidewalk.",
            "A woman is sitting at a desk and working on a laptop.",
            "A dog is running through a field of grass."
        ]
    
        for example in examples:
            print(f"Source: {example}")
            translation = translate_sentence(model, example, data_loader, device)
            print(f"Translation: {translation}\n")
def main():
    logger.info("Create Dataset Loaders")
    # Initialize the data loader class
    batch_size = 32
    data_loader = TranslationDataLoader(batch_size=batch_size)

    # Create datasets
    train_iter = Multi30k(split='train')
    valid_iter = Multi30k(split='valid')
    test_iter = Multi30k(split='test')

    # Create DataLoaders
    train_loader = DataLoader(train_iter, batch_size=data_loader.batch_size, collate_fn=data_loader.get_collate_fn())
    valid_loader = DataLoader(valid_iter, batch_size=data_loader.batch_size, collate_fn=data_loader.get_collate_fn())
    test_loader = DataLoader(test_iter, batch_size=data_loader.batch_size, collate_fn=data_loader.get_collate_fn())

    # Get an example batch
    example_batch = next(iter(train_loader))
    src, trg = example_batch
    logger.debug(f"batch size: {src.size()}")  # src.size() instead of example_batch[0].size()

    # Print special token indices
    special_token_indices = data_loader.get_special_token_indices()
    logger.debug("\nSpecial Token Indices:")
    for token, idx in special_token_indices.items():
        logger.debug(f"  {token}: {idx}")

    # Iterate over each sentence in the batch
    for i in range(min(5, src.size(0))):  # Changed to src.size(0) to iterate over batch_size
        src_tokens = [data_loader.vocab_transform[data_loader.src_language].lookup_token(token) for token in src[i] if token != data_loader.PAD_IDX]
        trg_tokens = [data_loader.vocab_transform[data_loader.trg_language].lookup_token(token) for token in trg[i] if token != data_loader.PAD_IDX]

        # Print the details for the current sequence
        logger.debug(
    f"""
    Example {i+1}
    Source Text: {' '.join(src_tokens[1:-1])}
    Source Tokens: {src_tokens}
    Source Tensor with Special Tokens: {src[i].tolist()}
    Example {i+1} Target Text: {' '.join(trg_tokens[1:-1])}
    Target Tokens: {trg_tokens}
    Target Tensor with Special Tokens: {trg[i].tolist()}
    """)
        
    src_vocab_size, trg_vocab_size = data_loader.get_vocab_size()

    device = 'cuda'
    model = Transformer(
        enc_voc_size = src_vocab_size,
        dec_vocal_size = trg_vocab_size,
        d_model=512,
        max_len=256,
        src_padding_token = data_loader.PAD_IDX,
        tgt_padding_token = data_loader.PAD_IDX,
        dim_feedforward = 2048,
        drop_prob =0.2,
        num_encoder_layers = 8,
        num_decoder_layers = 8,
        n_head = 8
    )
    optimizer = Adam(params=model.parameters(),
                 lr=1e-5,
                 weight_decay=5e-4,
                 eps=5e-9)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=0.9,
                                                 patience=10)
    criterion = nn.CrossEntropyLoss(ignore_index=data_loader.PAD_IDX)
    
    model = model.to(device)
    print(model)
    for epoch in range(30):
        print("----------------------------------------------------------------------")
        model.train()
        epoch_loss = 0
        for i, train_batch in tqdm(enumerate(train_loader), desc=f"training {epoch}/30"):
            optimizer.zero_grad()
            src, trg = train_batch
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:,:-1])
            output = output.view(-1, output.size(-1))
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= i
        print(f"Epoch {epoch+1}/{30}, Train Loss: {epoch_loss:.4f}")
        
        val_loss = evaluate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}/{30}, Validation Loss: {val_loss:.4f}")
        inference_examples(model, data_loader, device)
        
       
        
    
    

if __name__ == '__main__':
    main()