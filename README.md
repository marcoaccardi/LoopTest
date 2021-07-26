# LoopTest 
[![GitHub](https://img.shields.io/github/license/allenhung1025/loop-generation?label=license)](./LICENSE.md)
![GitHub issues](https://img.shields.io/github/issues/allenhung1025/loop-generation)
![GitHub Repo stars](https://img.shields.io/github/stars/allenhung1025/loop-generation)
* This is the official repository of **A Benchmarking Initiative for Audio-domain Music Generation using the FreeSound Loop Dataset** co-authored with [Paul Chen](https://paulyuchen.com/), [Arthur Yeh](http://yentung.com/) and my supervisor [Yi-Hsuan Yang](http://mac.citi.sinica.edu.tw/~yang/). The paper has been accepted by International Society for Music Information Retrieval Conference 2021.  
* [Demo Page](https://loopgen.github.io/).
* Will put on arxiv link upon release of the paper.
* We not only provided pretrained model to generate loops on your own but also provided scripts for you to evaluate the generated loops.
## Environment
```
$ conda env create -f environment.yml 
```
## Quick Start
* Generate loops from looperman pretrained model
* If you want to "style-mix" the loop, you can add the argument `--style_mixing`.
* There are two example scripts ([freesound](./generate_freesound.sh), [looperman](./generate_looperman.sh)) in the repo. 
``` bash
$ gdown --id 1GQpzWz9ycIm5wzkxLsVr-zN17GWD3_6K -O looperman_checkpoint.pt
$ CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt "looperman_checkpoint.pt" \
    --pics 2000 --data_path "./data/looperman" \
    --store_path "./generated_looperman_one_bar"
``` 
* Generate loops from freesound pretrained model
``` bash
$ gdown --id 197DMCOASEMFBVi8GMahHfRwgJ0bhcUND -O freesound_checkpoint.pt 
$ CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt "freesound_checkpoint.pt" \
    --pics 2000 --data_path "./data/freesound" \
    --store_path "./generated_freesound_one_bar"
``` 
## Benchmarking Freesound Loop Dataset
### Download dataset
### Training

``` bash
CUDA_VISIBLE_DEVICES=2 python train_drum.py \
    --size 64 --batch 8 --sample_dir [sample_dir] \
    --checkpoint_dir [checkpoint_dir] \
    [mel-spectrogram dataset from the proprocessing]
```

### Generate audio
### Evaluation
## Pretrained Checkpoint
* [Looperman pretrained model link](https://drive.google.com/file/d/1GQpzWz9ycIm5wzkxLsVr-zN17GWD3_6K/view?usp=sharing) 
* [Freesound pretrained model link](https://drive.google.com/file/d/197DMCOASEMFBVi8GMahHfRwgJ0bhcUND/view?usp=sharing)

## Preprocess the Loop Dataset
In the [preprocess](./preprocess) directory and modify some settings (e.g. data path) in the codes and run them with the following orders
``` bash
$ python trim_2_seconds.py # Cut loop into the single bar and stretch them to 2 second.
$ python extract_mel.py # Extract mel-spectrogram from 2-second audio.
$ python make_dataset.py 
$ python compute_mean_std.py 
```

## Vocoder
We use [MelGAN][melgan] as the vocoder. We trained the vocoder with looperman dataset and use the vocoder in generating freesound and looperman models.
The trained vocoder is in [melgan](./melgan) directory.

## Train the Model
``` bash
CUDA_VISIBLE_DEVICES=2 python train_drum.py \
    --size 64 --batch 8 --sample_dir [sample_dir] \
    --checkpoint_dir [checkpoint_dir] \
    [mel-spectrogram dataset from the proprocessing]
```
* checkpoint_dir stores model in the designated directory.
* sample_dir stores mel-spectrogram generated from the model.
* You should give the data directory in the end.
* There is an example training [script](./train.sh)

## Evaluation
### NDB_JS
* 2000 looperman melspectrogram [link](https://drive.google.com/file/d/1aFGPYlkkAysVBWp9VacHVk2tf-b4rLIh/view?usp=sharing)
    ``` bash
    $ cd evaluation/NDB_JS
    $ gdwon --id 1aFGPYlkkAysVBWp9VacHVk2tf-b4rLIh
    $ unzip looper_2000.zip # contain 2000 looperman mel-sepctrogram
    $ bash compute_ndb_js.sh ## you have to modify this script to evaluation your generated melspectrograms
    ```
### IS
* Short-Chunk CNN [checkpoint](./evaluation/IS/best_model.ckpt)
    ``` bash
    $ cd evaluation/IS
    $ bash compute compute_is_score.sh ## you have to modify this script to evaluation your generated melspectrograms
    ```
### FAD
* FAD looperman ground truth [link](./evaluation/FAD/looperman_2000.stats), follow the official [doc](fad) to download the code and the evaluation.

    ``` bash
    $ ls --color=never test_audio/background/*  > test.csv
    $ python -m frechet_audio_distance.create_embeddings_main --input_files test.csv --stats test.stats
    $ python -m frechet_audio_distance.compute_fad --background_stats ./looperman_2000.stats --test_stats test.stats
    ```

## References
The code comes heavily from the code below
* [StyleGAN2 from rosinality][stylegan2]
* [Official MelGAN repo][melgan] 
* [Official UNAGAN repo from ciaua][unagan].
* [Official Short Chunk CNN repo][cnn]
* [FAD official document][fad]

[fad]: https://github.com/google-research/google-research/tree/master/frechet_audio_distance
[cnn]: https://github.com/minzwon/sota-music-tagging-models
[stylegan2]: https://github.com/rosinality/stylegan2-pytorch
[unagan]: https://github.com/ciaua/unagan
[melgan]: https://github.com/descriptinc/melgan-neurips

## Citation
If you find this repo useful, please cite with the following information.
```
@inproceedings{ allenloopgen, 
	title={A Benchmarking Initiative for Audio-domain Music Generation using the {FreeSound Loop Dataset}},
	author={Tun-Min Hung and Bo-Yu Chen and Yen-Tung Yeh, and Yi-Hsuan Yang},
	booktitle = {Proc. Int. Society for Music Information Retrieval Conf.},
	year={2021},
}
```
