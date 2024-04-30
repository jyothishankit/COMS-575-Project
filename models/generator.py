from models.conformer import ConformerBlock
import torch
import torch.nn as nn
from dimensions import *

class DilatedDenseNet(nn.Module):
    def __init__(self, dth=4, input_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.dth = dth
        self.input_channels = input_channels
        self.padding = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.k_size = (self.twidth, 3)
        for i in range(self.dth):
            dil = 2**i
            padding_len = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, padding_len, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.input_channels * (i + 1),
                    self.input_channels,
                    kernel_size=self.k_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(input_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.input_channels))

    def forward(self, x):
        skip = x
        for i in range(self.dth):
            output = getattr(self, "pad{}".format(i + 1))(skip)
            output = getattr(self, "conv{}".format(i + 1))(output)
            output = getattr(self, "norm{}".format(i + 1))(output)
            output = getattr(self, "prelu{}".format(i + 1))(output)
            skip = torch.cat([output, skip], dim=1)
        return output


class DenseEncoder(nn.Module):
    def __init__(self, input_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv2d(input_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.d_dense = DilatedDenseNet(dth=4, input_channels=channels)
        self.conv_2d = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1d(x)
        x = self.d_dense(x)
        x = self.conv_2d(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conf = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=TIME_CONFORMER_HEADS,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, time, freq = x_in.size()
        x_time = x_in.permute(0, 3, 2, 1).contiguous().view(b * freq, time, c)
        x_time = self.time_conf(x_time) + x_time
        x_freq = x_time.view(b, freq, time, c).permute(0, 2, 1, 3).contiguous().view(b * time, freq, c)
        x_freq = self.freq_conformer(x_freq) + x_freq
        x_freq = x_freq.view(b, time, freq, c).permute(0, 3, 1, 2)
        return x_freq


class SPConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, k_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.padding1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.output_channels = output_channels
        self.conv2d = nn.Conv2d(
            input_channels, output_channels * r, kernel_size=k_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.padding1(x)
        output = self.conv2d(x)
        b_size, nchannels, H, W = output.shape
        output = output.view((b_size, self.r, nchannels // self.r, H, W))
        output = output.permute(0, 2, 3, 4, 1)
        output = output.contiguous().view((b_size, nchannels // self.r, H, -1))
        return output


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, output_channel=1):
        super(MaskDecoder, self).__init__()
        self.d_dense_block = DilatedDenseNet(dth=4, input_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1d = nn.Conv2d(num_channel, output_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(output_channel, affine=True)
        self.prelu = nn.PReLU(output_channel)
        self.final_conv = nn.Conv2d(output_channel, output_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.d_dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1d(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.d_dense_block = DilatedDenseNet(dth=4, input_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.d_dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(input_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, output_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        magnitude = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        phase_noisy = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        x_input = torch.cat([magnitude, x], dim=1)

        output_1 = self.dense_encoder(x_input)
        output_2 = self.TSCB_1(output_1)
        output_3 = self.TSCB_2(output_2)
        output_4 = self.TSCB_3(output_3)
        output_5 = self.TSCB_4(output_4)

        mask = self.mask_decoder(output_5)
        output_magnitude = mask * magnitude

        complex_output = self.complex_decoder(output_5)
        magnitude_real = output_magnitude * torch.cos(phase_noisy)
        magnitude_imag = output_magnitude * torch.sin(phase_noisy)
        combo_real = magnitude_real + complex_output[:, 0, :, :].unsqueeze(1)
        combo_imag = magnitude_imag + complex_output[:, 1, :, :].unsqueeze(1)

        return combo_real, combo_imag
