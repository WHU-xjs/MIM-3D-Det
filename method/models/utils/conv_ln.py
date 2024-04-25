from mmcv.cnn import ConvModule

# not used since GeneralizedLSSFPN not used
class ConvModuleLN(ConvModule):
    """extended module to support LN, auto permute before and after norm"""
    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                if self.norm_cfg['type'] == 'LN':
                    x = x.permute(0,2,3,1)
                    x = self.norm(x)
                    x = x.permute(0,3,1,2)
                else:
                    x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x
