checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dict_file = 'dict_printed_chinese_english_digits.txt'
label_convertor = dict(
    type='AttnConvertor',
    dict_file='dict_printed_chinese_english_digits.txt',
    with_unknown=True)
model = dict(
    type='SARNet',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='SAREncoder', enc_bi_rnn=False, enc_do_rnn=0.1, enc_gru=False),
    decoder=dict(
        type='ParallelSARDecoder',
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True),
    loss=dict(type='SARLoss'),
    label_convertor=dict(
        type='AttnConvertor',
        dict_file='dict_printed_chinese_english_digits.txt',
        with_unknown=True),
    max_seq_len=30)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 100
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=256,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
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
                height=48,
                min_width=48,
                max_width=256,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
                ])
        ])
]
dataset_type = 'OCRDataset'
train_prefix = '/home/will/ocr_data'
train_ann_file = '/home/will/ocr_data/train_list_.txt'
train = dict(
    type='OCRDataset',
    img_prefix='/home/will/ocr_data',
    ann_file='/home/will/ocr_data/train_list_.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
test_prefix = '/home/will/ocr_data'
test_ann_file = '/home/will/ocr_data/val_list_.txt'
test = dict(
    type='OCRDataset',
    img_prefix='/home/will/ocr_data',
    ann_file='/home/will/ocr_data/val_list_.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
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
                img_prefix='/home/will/ocr_data',
                ann_file='/home/will/ocr_data/train_list_.txt',
                loader=dict(
                    type='HardDiskLoader',
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
                height=48,
                min_width=48,
                max_width=256,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'text',
                    'valid_ratio'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='/home/will/ocr_data',
                ann_file='/home/will/ocr_data/val_list_.txt',
                loader=dict(
                    type='HardDiskLoader',
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
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=48,
                        min_width=48,
                        max_width=256,
                        keep_aspect_ratio=True,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'resize_shape',
                            'valid_ratio'
                        ])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='/home/will/ocr_data',
                ann_file='/home/will/ocr_data/val_list_.txt',
                loader=dict(
                    type='HardDiskLoader',
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
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=48,
                        min_width=48,
                        max_width=256,
                        keep_aspect_ratio=True,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'resize_shape',
                            'valid_ratio'
                        ])
                ])
        ]))
evaluation = dict(interval=1, metric='acc')
work_dir = '/home/will/gaoxinjian/mmocr/SAR/'
gpu_ids = range(0, 2)
