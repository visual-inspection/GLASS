"""
This file is the same as "main_glink_ce_blanked.py", but version "E" which
delivered the best results so far.
"""

from main import main


main([
                "--gpu", "3",
                "--seed", "0",
                "--test", "test", # "ckpt", # 
                
                "net",
                "-b", "wideresnet50",
                "-le", "layer2",
                "-le", "layer3",
                "--pretrain_embed_dimension", "1536",
                "--target_embed_dimension", "1536",
                "--patchsize", "3",
                "--meta_epochs", "640",
                "--eval_epochs", "1",
                "--dsc_layers", "2",
                "--dsc_hidden", "1024",
                "--pre_proj", "1",
                "--mining", "1",
                "--noise", "0.015",
                "--radius", "0.75",
                "--p", "0.5",
                "--step", "20",
                "--limit", "392",
                
                "dataset",
                "--distribution", "0", # "1", # 1==judge
                "--mean", "0.5",
                "--std", "0.1",
                "--fg", "1",
                "--rand_aug", "1",
                "--batch_size", "8", # 8
                "--resize", "160", # 288
                "--imagesize", "160", # 288
                "-d", "ce_blanked",
                
                "glink",
                "/home/sehoaa/GLASS-repl_a/glink",
                "/space/sehoaa/dataset_dtd/images"
            ])