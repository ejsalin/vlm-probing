import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

import pickle

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        self.model_name = self.hparams.config["load_path"].split('/')[-1].split('.')[0]
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if "nlvr2" in self.hparams.config["load_path"]:
                self.token_type_embeddings = nn.Embedding(3, hs)
                emb_data = state_dict['text_embeddings.token_type_embeddings.weight']
                #token_type_embeddings = nn.Embedding(2, hs)
                #token_type_embeddings.apply(objectives.init_weights)
                #token_type_embeddings.weight.data[0, :] = emb_data.data[0, :]
                #token_type_embeddings.weight.data[1, :] = emb_data.data[1, :]
                #del state_dict['text_embeddings.token_type_embeddings.weight']
                #state_dict['text_embeddings.token_type_embeddings.weight'] = None
            self.load_state_dict(state_dict, strict=False)
            if "nlvr2" in self.hparams.config["load_path"]:
                emb_data = self.token_type_embeddings.weight.data
                self.token_type_embeddings = nn.Embedding(2, hs)
                self.token_type_embeddings.apply(objectives.init_weights)
                self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
                self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]




    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        #print("Doing inference")
        
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        #cls_feats = self.pooler(x)

        if 'color' in batch['table_name'][0]:
            output_feat_index = text_ids[0].tolist().index(103)
            feat = text_feats[:, output_feat_index] 
        elif 'tag' in  batch['table_name'][0]:
            feat = text_feats
        else:
            #print('Ici')
            feat = x[:, 0]

        ret = {
            #"text_feats": text_feats,
            #"image_feats": image_feats,
            #"cls_feats": cls_feats,
            "raw_cls_feats": feat,
            #"image_labels": image_labels,
            #"image_masks": image_masks,
            #"text_labels": text_labels,
            "text_ids": text_ids,
            #"text_masks": text_masks,
            #"patch_index": patch_index,
        }
        #print(mask_text, do_mlm)
        #exit()
        #print("batch", batch)
        #print("ret", ret)
        #exit()

        return ret

    def forward(self, batch):
        #print("Doing forward")
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        
        #print("test step")
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()
        #print("batch", batch_idx, batch, output)
        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))
        
        #return ret
        output['table_name'] = batch['table_name']
        output['raw_index'] = batch['raw_index']
        output['target'] = batch['target']
        return  output
        


    def test_epoch_end(self, outs):
        #print("outs", outs[1])
        dico_save = dict()
        for output_i in outs:
            prob_task = output_i['table_name'][0]
            if prob_task not in dico_save.keys():
                dico_save[prob_task] = [ output_i]
            else:
                dico_save[prob_task].append(output_i)
        for key in dico_save.keys():
            with open(f"aaai_probing_outputs/{self.model_name}_{key}.pickle", 'wb') as outfile:
                pickle.dump(dico_save[key], outfile)
        
        print("test epoch end")
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
