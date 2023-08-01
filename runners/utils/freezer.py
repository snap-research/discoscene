# python3.7
"""Contains the class for fixing parameters."""

__all__ = ['Freezer']


class Freezer(object):
    """Defines the freezer.

    `Freeze` means to fix some certain parameters such that they cannot get
    updated during training. To save time, in this implementation, frozen is
    achieved by directly converting those parameters to buffers.
    """

    @staticmethod
    def param_to_buffer(module, name):
        """Converts the parameter in `module` to a buffer with the same `name`.

        Args:
            module: `nn.Module`, the root module to start the params searching.
            name: `str`, the name of parameters, where names of sub-modules are
                joined by `.` (e.g., 'synthesis.layer5.style.weight').
        """
        split_name = name.split('.')
        module_name_hierarchy = split_name[:-1]
        param_name = split_name[-1]
        tgt_module = module
        for module_name in module_name_hierarchy:
            tgt_module = getattr(tgt_module, module_name)
        param_data = getattr(tgt_module, param_name).data
        delattr(tgt_module, param_name)
        tgt_module.register_buffer(name=param_name, tensor=param_data)

    @staticmethod
    def freeze_by_keywords(module, keywords=None, exclusive_keywords=None):
        """Freezes parameters that matched by the given keywords.

        Args:
            module: `nn.Module`, the root module to start the params searching.
            keywords: `str`, the keys to match target parameters. Multiple keys
                can be provided with comma as the separator. If set to `*`,
                the entire `module` will be frozen. If set to `None`, nothing
                will be frozen. (default: None)
            exclusive_keywords: `str`, the keywords to be excluded for target
                parameters. Multiple keys can be provided with comma as the
                separator. If set to `None`, this method will check all
                parameters inside the module. (default: None)

        Examples:
            >>> # Freeze the mapping network of StyleGAN2:
            >>> Freezer.freeze_by_keywords(StyleGAN2Generator, 'mapping')
            >>> # Another implementation:
            >>> Freezer.freeze_by_keywords(StyleGAN2Generator, '*',
                                           exclusive_keywords='synthesis')

            >>> # Freeze the mapping network and affine layers of StyleGAN2:
            >>> Freezer.freeze_by_keywords(StyleGAN2Generator, 'mapping, style')
        """
        if not keywords:  # Shortcut of freezing nothing.
            return

        # Get parameter name list.
        param_list = Freezer.get_module_param_names(module, exclusive_keywords)

        if keywords == '*':  # Shortcut of freezing everything.
            for name in param_list:
                Freezer.param_to_buffer(module, name)
            return

        # Freeze parameters that contains any keyword.
        keywords = keywords.replace(' ', '').split(',')
        for name in param_list:
            if any(keyword in name for keyword in keywords):
                Freezer.param_to_buffer(module, name)

    @staticmethod
    def get_module_param_names(module, exclusive_keywords=None):
        """Gets all parameter names not containing any of exclusive keywords.

        Args:
            module: `nn.Module`, the root module to start the params searching.
            exclusive_keywords: `str`, the words to be excluded for target
                parameters. Multiple keys can be provided with comma as the
                separator. If set to `None`, this method returns all parameter
                names inside the module. (default: None)

        Returns:
            A list of parameter names filtered out by `exclusive_keywords`.

        Examples:
            >>> # Get all parameters of StyleGAN discriminator except the task
            >>> # head.
            >>> param_names = Freezer.get_module_param_names(
                    StyleGANDiscriminator, exclusive_keywords='output')
        """
        if not exclusive_keywords:  # Shortcut without filtering.
            return [name for name, _ in module.named_parameters()]

        # Filter parameters by name.
        exclusive_keywords = exclusive_keywords.replace(' ', '').split(',')
        return [name for name, _ in module.named_parameters()
                if all(key not in name for key in exclusive_keywords)]
