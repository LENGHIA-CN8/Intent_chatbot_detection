import os
from sentence_transformers import SentenceTransformer, models, ParallelSentencesDataset

def train(teacher_ckpt = 'all-mpnet-base-v2', student_ckpt = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', max_sentences_per_language = 1000000, train_max_sentence_length = 250, epochs = 1 ):
    
    warmup_steps = int(len(loader) * epochs * 0.1)
    ## Intialize teacher model
    teacher = SentenceTransformer(teacher_ckpt)
    mpnet = teacher[0]
    pooler = teacher[1]
    teacher = SentenceTransformer(modules=[mpnet, pooler])
    
    ## Initialize student model
    student = SentenceTransformer(student_ckpt)
    
    data = ParallelSentencesDataset(student_model=student, teacher_model=teacher, batch_size=32, use_embedding_cache=True)
    
    data.load_data('./envi_data/phoMT-train.tsv.gz', max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length)

    student.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path='./multi-phoMT',
        optimizer_params={'lr': 2e-5, 'eps': 1e-6},
        save_best_model=True,
        show_progress_bar=False
    )
    
if __name__ = "__main__":
    train()