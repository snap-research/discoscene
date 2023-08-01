# python3.7
"""Configuration for fine-tuning StyleGAN2."""

from .stylegan2_config import StyleGAN2Config

__all__ = ['StyleGAN2FineTuneConfig']


class StyleGAN2FineTuneConfig(StyleGAN2Config):
    """Defines the configuration for fine-tuning StyleGAN2."""

    name = 'stylegan2_finetune'
    hint = 'Fine-tune a StyleGAN2 model by freezing selected parameters.'
    info = '''
It is possible to fine-tune a StyleGAN2 model by partially freezing the
parameters of the generator and the discriminator. This trick is commonly used
when the training data is limited, to prevent overfitting.

For the generator, consisting of a mapping network and a synthesis network,
users can use `freeze_g_mapping_layers` and `freeze_g_synthesis_blocks` to
control the behavior of these two parts independently. As for a particular layer
in the synthesis network, it contains an affine layer (fully-connected layer) to
learn per-layer style, a convolutional layer, a noise modulation operation, and
a ToRGB layer (only after each block). Users can use `freeze_g_affine`,
`freeze_g_conv`, `freeze_g_noise`, `freeze_g_torgb` to control these four parts,
separately. Note that, the embedding layer for conditional synthesis, and the
learnable constant tensor for synthesis network, should be separately
configured.

For the discriminator, consisting of a backbone and a bi-classification head,
users can use `freeze_d_blocks` and `freeze_d_adv_head` to control the behavior
of these two parts independently. Note that, the embedding layer for conditional
synthesis, and the input layer of the backbone, should be separately configured.
'''

    @classmethod
    def get_options(cls):
        options = super().get_options()

        options['Generator fine-tuning settings'].extend([
            cls.command_option(
                '--freeze_g_embedding', type=cls.bool_type, default=False,
                help='Whether to freeze the embedding layer in the generator '
                     'for conditional synthesis.'),
            cls.command_option(
                '--freeze_g_mapping_layers', type=cls.index_type, default=None,
                help='Indices of layers in the mapping network to freeze. Use '
                     'comma to join multiple indices.'),
            cls.command_option(
                '--freeze_g_const', type=cls.bool_type, default=False,
                help='Whether to freeze the initial learnable constant.'),
            cls.command_option(
                '--freeze_g_synthesis_blocks', type=cls.index_type,
                default=None,
                help='Indices of blocks in the synthesis network to freeze. '
                     'Use comma to join multiple indices.'),
            cls.command_option(
                '--freeze_g_affine', type=cls.bool_type, default=False,
                help='Whether to freeze the style affine transformations.'),
            cls.command_option(
                '--freeze_g_conv', type=cls.bool_type, default=False,
                help='Whether to freeze the convolution layers.'),
            cls.command_option(
                '--freeze_g_noise', type=cls.bool_type, default=False,
                help='Whether to freeze the noise modulation parameters.'),
            cls.command_option(
                '--freeze_g_torgb_affine', type=cls.bool_type, default=False,
                help='Whether to freeze the style affine transformations '
                     'within the ToRGB layers.'),
            cls.command_option(
                '--freeze_g_torgb', type=cls.bool_type, default=False,
                help='Whether to freeze the ToRGB convolutional layers.'),
            cls.command_option(
                '--freeze_g_keywords', type=str, default=None,
                help='Additional keywords used to select the parameters of the '
                     'generator that should be frozen. Use comma to join '
                     'multiple keys.')
        ])

        options['Discriminator fine-tuning settings'].extend([
            cls.command_option(
                '--freeze_d_embedding', type=cls.bool_type, default=False,
                help='Whether to freeze the embedding layer in the '
                     'discriminator for conditional synthesis.'),
            cls.command_option(
                '--freeze_d_mapping_layers', type=cls.index_type, default=None,
                help='Indices of layers in the mapping network of the '
                     'discriminator to freeze. Use comma to join multiple '
                     'indices.'),
            cls.command_option(
                '--freeze_d_blocks', type=cls.index_type, default=None,
                help='Indices of blocks in the discriminator to freeze. Use '
                     'comma to join multiple indices.'),
            cls.command_option(
                '--freeze_d_input', type=cls.bool_type, default=False,
                help='Whether to freeze the input layer of the to-freeze'
                     'blocks of the discriminator backbone.'),
            cls.command_option(
                '--freeze_d_adv_head', type=cls.bool_type, default=False,
                help='Whether to freeze the bi-classification task head.'),
            cls.command_option(
                '--freeze_d_keywords', type=str, default=None,
                help='Additional keywords used to select the parameters of the '
                     'discriminator that should be frozen. Use comma to join '
                     'multiple keys.')
        ])

        return options

    @classmethod
    def get_recommended_options(cls):
        recommended_opts = super().get_recommended_options()
        recommended_opts.extend([
            'freeze_g_embedding', 'freeze_g_mapping_layers', 'freeze_g_const',
            'freeze_g_synthesis_blocks', 'freeze_g_affine', 'freeze_g_conv',
            'freeze_g_noise', 'freeze_g_torgb_affine', 'freeze_g_torgb',
            'freeze_g_keywords', 'freeze_d_embedding',
            'freeze_d_mapping_layers', 'freeze_d_blocks', 'freeze_d_input',
            'freeze_d_adv_head', 'freeze_d_keywords'
        ])
        return recommended_opts

    def parse_options(self):
        super().parse_options()

        # Get parameters to freeze in generator.
        freeze_g_embedding = self.args.pop('freeze_g_embedding')
        freeze_g_mapping_layers = self.args.pop('freeze_g_mapping_layers')
        freeze_g_const = self.args.pop('freeze_g_const')
        freeze_g_synthesis_blocks = self.args.pop('freeze_g_synthesis_blocks')
        freeze_g_affine = self.args.pop('freeze_g_affine')
        freeze_g_conv = self.args.pop('freeze_g_conv')
        freeze_g_noise = self.args.pop('freeze_g_noise')
        freeze_g_torgb_affine = self.args.pop('freeze_g_torgb_affine')
        freeze_g_torgb = self.args.pop('freeze_g_torgb')
        freeze_g_keywords = self.args.pop('freeze_g_keywords')

        g_freeze_param_list = []
        # Categorical embedding.
        if freeze_g_embedding:
            g_freeze_param_list.append('mapping.embedding')
        # Mapping network.
        freeze_g_mapping_layers = freeze_g_mapping_layers or list()
        for idx in freeze_g_mapping_layers:
            g_freeze_param_list.append(f'mapping.dense{idx}.')
        # Learnable constant tensor.
        if freeze_g_const:
            g_freeze_param_list.append('synthesis.early_layer.const')
        # Synthesis network.
        freeze_g_synthesis_blocks = freeze_g_synthesis_blocks or list()
        for block_idx in freeze_g_synthesis_blocks:
            # Handle each convolutional layer.
            if block_idx != 0:
                layer_indices = [block_idx * 2 - 1, block_idx * 2]
            else:
                layer_indices = [0]
            for layer_idx in layer_indices:
                if freeze_g_affine:
                    g_freeze_param_list.append(
                        f'synthesis.layer{layer_idx}.style')
                if freeze_g_conv:
                    g_freeze_param_list.append(
                        f'synthesis.layer{layer_idx}.weight')
                    g_freeze_param_list.append(
                        f'synthesis.layer{layer_idx}.bias')
                if freeze_g_noise:
                    g_freeze_param_list.append(
                        f'synthesis.layer{layer_idx}.noise_strength')
            # Handle each residual layer.
            if freeze_g_conv:
                g_freeze_param_list.append(f'synthesis.residual{block_idx}.')
            # Handle each ToRGB layers.
            if freeze_g_torgb_affine:
                g_freeze_param_list.append(f'synthesis.output{block_idx}.style')
            if freeze_g_torgb:
                g_freeze_param_list.append(
                    f'synthesis.output{block_idx}.weight')
                g_freeze_param_list.append(f'synthesis.output{block_idx}.bias')
        # Additional keywords.
        if freeze_g_keywords:
            for keyword in freeze_g_keywords.replace(' ', '').split(','):
                g_freeze_param_list.append(keyword)

        self.config.models.generator.update(
            freeze_keywords=','.join(g_freeze_param_list)
        )

        # Get parameters to freeze in discriminator.
        freeze_d_embedding = self.args.pop('freeze_d_embedding')
        freeze_d_mapping_layers = self.args.pop('freeze_d_mapping_layers')
        freeze_d_blocks = self.args.pop('freeze_d_blocks')
        freeze_d_input = self.args.pop('freeze_d_input')
        freeze_d_adv_head = self.args.pop('freeze_d_adv_head')
        freeze_d_keywords = self.args.pop('freeze_d_keywords')

        d_freeze_param_list = []
        # Categorical embedding.
        if freeze_d_embedding:
            d_freeze_param_list.append('embedding')
        # Mapping network.
        freeze_d_mapping_layers = freeze_d_mapping_layers or list()
        for idx in freeze_d_mapping_layers:
            d_freeze_param_list.append(f'mapping{idx}.')
        # Backbone.
        freeze_d_blocks = freeze_d_blocks or list()
        for block_idx in freeze_d_blocks:
            if freeze_d_input:
                d_freeze_param_list.append(f'input{block_idx}.')
            d_freeze_param_list.append(f'layer{block_idx * 2}.')
            d_freeze_param_list.append(f'layer{block_idx * 2 + 1}.')
            d_freeze_param_list.append(f'residual{block_idx}')
        if freeze_d_adv_head:
            d_freeze_param_list.append('output.')
        # Additional keywords.
        if freeze_d_keywords:
            for keyword in freeze_d_keywords.replace(' ', '').split(','):
                d_freeze_param_list.append(keyword)

        self.config.models.discriminator.update(
            freeze_keywords=','.join(d_freeze_param_list)
        )
