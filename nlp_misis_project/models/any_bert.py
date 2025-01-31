import plotly.graph_objects as plotly_go
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import ConfusionMatrix, Accuracy, F1Score
from transformers import get_linear_schedule_with_warmup, AutoConfig, \
    AutoModelForSequenceClassification


class AnyBertClassifier(pl.LightningModule):
    def __init__(self, lr: float = 2e-5, base_model: str = 'roberta-base', warmup_steps: int = 0, num_labels: int = 2,
                 token_embedding_size: int | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=['lr'])
        self.lr = lr

        self.config = AutoConfig.from_pretrained(base_model, num_labels=num_labels)
        self.any_bert_model = AutoModelForSequenceClassification.from_pretrained(base_model, config=self.config,
                                                                                 ignore_mismatched_sizes=True)
        if token_embedding_size:
            self.any_bert_model.resize_token_embeddings(token_embedding_size)
        self.confusion_matrix = ConfusionMatrix(task='binary', num_labels=num_labels)
        self.accuracy = Accuracy(task='binary', num_labels=num_labels)
        self.f1_score = F1Score(task='binary', num_labels=num_labels, average='macro')

    def update_token_embedding_size(self, token_embedding_size: int):
        self.any_bert_model.resize_token_embeddings(token_embedding_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor):
        return self.any_bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx: int):
        labels, input_ids, attention_mask = batch
        # labels = labels.squeeze(-1).long()
        output = self(input_ids, attention_mask, labels)
        self.log('train_ce_loss', output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx: int):
        labels, input_ids, attention_mask = batch
        # labels = labels.squeeze(-1).long()
        output = self(input_ids, attention_mask, labels)
        self.log('val_ce_loss', output.loss, prog_bar=True)
        self.confusion_matrix.update(torch.argmax(output.logits, dim=-1), labels)
        self.accuracy.update(torch.argmax(output.logits, dim=-1), labels)
        self.f1_score.update(torch.argmax(output.logits, dim=-1), labels)
        return output.loss

    def on_validation_epoch_end(self) -> None:
        cm = self.confusion_matrix.compute().cpu().numpy()
        cm_text = cm.astype('str')
        fig = plotly_go.Figure(
            data=plotly_go.Heatmap(
                z=cm,
                x=['Не сарказм', 'Сарказм'],
                y=['Не сарказм', 'Сарказм'],
                showscale=True,
                colorscale="Viridis",
                text=cm_text,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Предсказанные классы',
            yaxis_title='Реальные классы',
        )
        wandb.log({
            'generated_confusion_matrix': [
                wandb.Html(fig.to_html())],
        })

        self.log('val_epoch_acc', self.accuracy.compute(), prog_bar=True)
        self.log('val_epoch_f1', self.f1_score.compute(), prog_bar=True)

        self.confusion_matrix.reset()
        self.accuracy.reset()
        self.f1_score.reset()

    def test_step(self, batch, batch_idx: int):
        labels, input_ids, attention_mask = batch
        # labels = labels.squeeze(-1).long()
        output = self(input_ids, attention_mask, labels)
        self.log('test_ce_loss', output.loss, prog_bar=True)
        self.accuracy.update(torch.argmax(output.logits, dim=-1), labels)
        self.f1_score.update(torch.argmax(output.logits, dim=-1), labels)
        return output.loss

    def on_test_epoch_end(self) -> None:
        self.log('test_epoch_acc', self.accuracy.compute(), prog_bar=True)
        self.log('test_epoch_f1', self.f1_score.compute(), prog_bar=True)
        self.accuracy.reset()
        self.f1_score.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
