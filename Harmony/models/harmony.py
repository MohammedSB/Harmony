import torch
from torchvision import models as torchvision_models

import Harmony.models.vision_transformer as vits
from .generative import GenerativePath
from .discriminative import DiscriminativePath

class Harmony(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.meta = vars(args)
        self.objective = args.objective

        # define the model arch (i.e. dino, dino+mae, harmony)
        self.define_arch()    

    def define_arch(self):

        # define the image encoder
        try: 
            self.meta['arch'] in vits.__dict__.keys()
            self.image_encoder = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                drop_path_rate=self.meta['drop_path_rate'] if hasattr(self.meta, 'drop_path_rate') else 0,  # stochastic depth
                return_all_tokens=True if "ibot" in self.objective else False,
                masked_im_modeling=self.meta['use_masked_im_modeling']
            )
        except:
            raise Exception(f"Unknow arch: {self.meta['arch']}")

        self.meta['embed_dim'] = self.image_encoder.embed_dim
       
        # initialize the variables
        self.is_discriminative = False
        self.is_generative = False

        if "dino" in self.objective or "ibot" in self.objective:
            self.discriminative_path = DiscriminativePath(image_encoder=self.image_encoder, meta=self.meta).cuda()
            self.is_discriminative = True

        if "mae" in self.objective:
            self.generative_path = GenerativePath(image_encoder=self.image_encoder, meta=self.meta).cuda()
            self.is_generative = True

    def forward(self, images, epoch, masks):
        loss = torch.tensor([0.0]).to(self.meta['gpu'])
        outputs = {"loss": loss,
                   "disc_loss": torch.zeros(1),
                   "gen_loss": torch.zeros(1)}

        if self.is_discriminative:
            output = self.discriminative_path(images[1:], epoch, masks)
            
            outputs["teacher_output"] = output["teacher_output"]
            outputs["teacher_output"] = output["student_output"]
            outputs["disc_loss"] = (output["loss"] * self.meta["disc_weight"])
            outputs["loss"] += (output["loss"] * self.meta["disc_weight"])

        if self.is_generative:
            output = self.generative_path(images, reconstruct_global_crops=self.meta['reconstruct_global_crops']) 
            
            outputs["pred"] = output["output"]
            outputs["mask"] = output["mask"]
            outputs["gen_loss"] = output["loss"]
            outputs["loss"] += output["loss"]

        return outputs