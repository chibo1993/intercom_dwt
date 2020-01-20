# Using the Discrete Wavelet Transform, convert the chunks of samples
# intro chunks of Wavelet coefficients (coeffs).
#
# The coefficients require more bitplanes than the original samples,
# but most of the energy of the samples of the original chunk tends to
# be into a small number of coefficients that are localized, usually
# in the low-frequency subbands:
#
# (supposing a chunk with 1024 samples)
#
# Amplitude
#     |       +                      *
#     |   *     *                  *
#     | *        *                *
#     |*          *             *
#     |             *       *
#     |                 *
#     +------------------------------- Time
#     0                  ^        1023 
#                |       |       
#               DWT  Inverse DWT 
#                |       |
#                v
# Amplitude
#     |*
#     |
#     | *
#     |  **
#     |    ****
#     |        *******
#     |               *****************
#     +++-+---+------+----------------+ Frequency
#     0                            1023
#     ^^ ^  ^     ^           ^
#     || |  |     |           |
#     || |  |     |           +--- Subband H1 (16N coeffs)
#     || |  |     +--------------- Subband H2 (8N coeffs)
#     || |  +--------------------- Subband H3 (4N coeffs)
#     || +------------------------ Subband H4 (2N coeffs)
#     |+-------------------------- Subband H5 (N coeffs)
#     +--------------------------- Subband L5 (N coeffs)
#
# (each channel must be transformed independently)
#
# This means that the most-significant bitplanes, for most chunks
# (this depends on the content of the chunk), should have only bits
# different of 0 in the coeffs that belongs to the low-frequency
# subbands. This will be exploited in a future issue.
#
# The straighforward implementation of this issue is to transform each
# chun without considering the samples of adjacent
# chunks. Unfortunately this produces an error in the computation of
# the coeffs that are at the beginning and the end of each subband. To
# compute these coeffs correctly, the samples of the adjacent chunks
# i-1 and i+1 should be used when the chunk i is transformed:
#
#   chunk i-1     chunk i     chunk i+1
# +------------+------------+------------+
# |          OO|OOOOOOOOOOOO|OO          |
# +------------+------------+------------+
#
# O = sample
#
# (In this example, only 2 samples are required from adajact chunks)
#
# The number of ajacent samples depends on the Wavelet
# transform. However, considering that usually a chunk has a number of
# samples larger than the number of coefficients of the Wavelet
# filters, we don't need to be aware of this detail if we work with
# chunks.

import struct
import matplotlib.pyplot as plt
import pywt as wt
import numpy as np
from intercom import Intercom
from intercom_empty import Intercom_empty

if __debug__:
    import sys

class Intercom_DWT(Intercom_empty):

    def init(self, args):
        Intercom_empty.init(self, args)
        self.skipped_bitplanes = [0]*self.cells_in_buffer

        # Number of levels of the DWT
        levels = 4
        # Wavelet used Biorthogonal 3.5
        #Referencia http://wavelets.pybytes.com/wavelet/bior3.5/
        wavelet = 'bior3.5'
        #padding = "symmetric"
        padding = "periodization"

    # Energy of the signal x
    def energy(x):
        return np.sum(x*x)/len(x)

    def send_bitplane(self, indata, bitplane_number):
        bitplane = (indata[:, bitplane_number%self.number_of_channels] >> bitplane_number//self.number_of_channels) & 1
        if np.any(bitplane): 
            bitplane = bitplane.astype(np.uint8)
            bitplane = np.packbits(bitplane)
            message = struct.pack(self.packet_format, self.recorded_chunk_number, bitplane_number, self.received_bitplanes_per_chunk[(self.played_chunk_number+1) % self.cells_in_buffer]+1, *bitplane)
            self.sending_sock.sendto(message, (self.destination_IP_addr, self.destination_port))
        else:
            self.skipped_bitplanes[self.recorded_chunk_number % self.cells_in_buffer] += 1
        # Get the number of wavelet coefficients to get the number of samples
        shapes = wt.wavedecn_shapes((bitplane,), wavelet)

        #Devuelve valores espaciados uniformemente dentro de un intervalo dado.
        #Los valores se generan dentro del intervalo medio abierto (en otras palabras, el intervalo que incluye inicio pero excluye detención ). 
        #Para argumentos enteros, la función es equivalente a la función de rango incorporada de Python , pero devuelve un ndarray en lugar de una lista.
        #[start, stop)
                      #(inicio,parada, paso)
        sample = np.arange(0, bitplane, 1)

        #Agregar una subtrama a la figura actual.
        #Contenedor Figure.add_subplotcon una diferencia de comportamiento explicado en la sección de notas.
        fig, axs = plt.subplots(bitplane//skipped_bitplanes,1, sharex=True)

        for i in range(0, bitplane,skipped_bitplanes):
            axs[i//skipped_bitplanes].set_ylim(-bitplane, bitplane)
            axs[i//skipped_bitplanes].grid(True)
            #Dibujamos en el eje de las cordenadas las muestras y la amplitud.
        axs[bitplane//skipped_bitplanes-1].set_xlabel('sample')
        axs[bitplane//skipped_bitplanes-2].set_ylable('Amplitud')
        
        print("Coefficient\t   Energy")

        zeros = np.zeros(bitplane)
        coeffs = wt.wavedec(zeros, wavelet=wavelet, level=levels, mode=padding)
        arr, coeff_slices = wt.coeffs_to_array(coeffs)
        for i in range(0,number_of_samples,skip):

            arr[i] = bitplane # i is the coeff different from 0
            coeffs_from_arr = wt.array_to_coeffs(arr, coeff_slices, output_format="wavedec")
            samples = wt.waverec(coeffs_from_arr, wavelet=wavelet, mode=padding)
            arr[i] = 0
            print("       %4d" % i, "\t", "%8.2f" % energy(samples))
            axs[i//skip].plot(sample, samples)

        plt.show()

    def send(self, indata):
        signs = indata & 0x8000
        magnitudes = abs(indata)
        indata = signs | magnitudes
        self.NOBPTS = int(0.75*self.NOBPTS + 0.25*self.NORB)
        self.NOBPTS += self.skipped_bitplanes[(self.played_chunk_number+1) % self.cells_in_buffer]
        self.skipped_bitplanes[(self.played_chunk_number+1) % self.cells_in_buffer] = 0
        self.NOBPTS += 1
        if self.NOBPTS > self.max_NOBPTS:
            self.NOBPTS = self.max_NOBPTS
        last_BPTS = self.max_NOBPTS - self.NOBPTS - 1
        #self.send_bitplane(indata, self.max_NOBPTS-1)
        #self.send_bitplane(indata, self.max_NOBPTS-2)
        #for bitplane_number in range(self.max_NOBPTS-3, last_BPTS, -1):
        for bitplane_number in range(self.max_NOBPTS-1, last_BPTS, -1):
            self.send_bitplane(indata, bitplane_number)
        self.recorded_chunk_number = (self.recorded_chunk_number + 1) % self.MAX_CHUNK_NUMBER

if __name__ == "__main__":
    intercom = Intercom_DWT()
    parser = intercom.add_args()
    args = parser.parse_args()
    intercom.init(args)
    intercom.run()
