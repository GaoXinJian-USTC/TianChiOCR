checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True, lower=True)
model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SatrnEncoder',
        n_layers=12,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        n_position=100,
        d_inner=2048,
        dropout=0.1),
    decoder=dict(
        type='TFDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=2048,
        d_k=64,
        d_v=64),
    loss=dict(type='TFLoss'),
    label_convertor=dict(
        type='AttnConvertor',
        dict_type='DICT90',
        with_unknown=True,
        lower=True),
    max_seq_len=25)
optimizer = dict(type='Adam', lr=0.0003)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 6
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(
        type='NormalizeOCR',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=100,
                max_width=100,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape'
                ])
        ])
]
dataset_type = 'OCRDataset'
train_prefix = 'data/mixture/benchmark/'
train_img_prefix1 = 'data/mixture/benchmark/SynthText/synthtext/SynthText_patch_horizontal'
train_img_prefix2 = 'data/mixture/benchmark/mjsynth/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = 'data/mixture/benchmark/SynthText/label.lmdb'
train_ann_file2 = 'data/mixture/benchmark/mjsynth/label.lmdb'
train1 = dict(
    type='OCRDataset',
    img_prefix=
    'data/mixture/benchmark/SynthText/synthtext/SynthText_patch_horizontal',
    ann_file='data/mixture/benchmark/SynthText/label.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train2 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/benchmark/mjsynth/mnt/ramdisk/max/90kDICT32px',
    ann_file='data/mixture/benchmark/mjsynth/label.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
test_prefix = 'data/mixture/benchmark/'
test_img_prefix1 = 'data/mixture/benchmark/IIIT5K/test'
test_img_prefix2 = 'data/mixture/benchmark/SVT/image'
test_img_prefix5 = 'data/mixture/benchmark/SVTP/IMG'
test_img_prefix6 = 'data/mixture/benchmark/CUTE80/CUTE80'
test_ann_file1 = 'data/mixture/benchmark/IIIT5K/test_label.txt'
test_ann_file2 = 'data/mixture/benchmark/SVT/test_label.txt'
test_ann_file5 = 'data/mixture/benchmark/SVTP/test_label.txt'
test_ann_file6 = 'data/mixture/benchmark/CUTE80/test_label.txt'
test1 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/benchmark/IIIT5K/test',
    ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test2 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/benchmark/SVT/image',
    ann_file='data/mixture/benchmark/SVT/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test3 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/benchmark/IIIT5K/test',
    ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test4 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/benchmark/IIIT5K/test',
    ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test5 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/benchmark/SVTP/IMG',
    ann_file='data/mixture/benchmark/SVTP/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test6 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/benchmark/CUTE80/CUTE80',
    ann_file='data/mixture/benchmark/CUTE80/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                'data/mixture/benchmark/SynthText/synthtext/SynthText_patch_horizontal',
                ann_file='data/mixture/benchmark/SynthText/label.lmdb',
                loader=dict(
                    type='LmdbLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix=
                'data/mixture/benchmark/mjsynth/mnt/ramdisk/max/90kDICT32px',
                ann_file='data/mixture/benchmark/mjsynth/label.lmdb',
                loader=dict(
                    type='LmdbLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=100,
                max_width=100,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'text',
                    'valid_ratio', 'resize_shape'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/IIIT5K/test',
                ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/SVT/image',
                ann_file='data/mixture/benchmark/SVT/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/IIIT5K/test',
                ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/IIIT5K/test',
                ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/SVTP/IMG',
                ann_file='data/mixture/benchmark/SVTP/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/CUTE80/CUTE80',
                ann_file='data/mixture/benchmark/CUTE80/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=32,
                        min_width=100,
                        max_width=100,
                        keep_aspect_ratio=False,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'resize_shape'
                        ])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/IIIT5K/test',
                ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/SVT/image',
                ann_file='data/mixture/benchmark/SVT/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/IIIT5K/test',
                ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/IIIT5K/test',
                ann_file='data/mixture/benchmark/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/SVTP/IMG',
                ann_file='data/mixture/benchmark/SVTP/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/benchmark/CUTE80/CUTE80',
                ann_file='data/mixture/benchmark/CUTE80/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=32,
                        min_width=100,
                        max_width=100,
                        keep_aspect_ratio=False,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'resize_shape'
                        ])
                ])
        ]))
evaluation = dict(interval=1, metric='acc')
work_dir = '/home/will/gaoxinjian/mmocr/SATRN/'
gpu_ids = range(0, 2)
