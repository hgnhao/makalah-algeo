import numpy as np
import pandas as pd
import os
import math

# Fungsi untuk menghitung entropy dari nilai singular
def calc_entropy(singular_values):
    probabilities = singular_values / np.sum(singular_values)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Fungsi utama
if __name__ == "__main__":
    # Nama file CSV
    file_name = input("Masukkan nama file CSV (contoh: data_liga.csv): ")
    output_file = input("Masukkan nama file output (contoh: hasil_svd.txt): ")

    input_path = os.path.join('data', file_name)
    output_path = os.path.join('output', output_file)
    
    try:
        # Membaca data CSV
        data = pd.read_csv(input_path)
        print("\nData Liga:\n", data)

        # Memastikan hanya kolom numerik yang diambil untuk SVD (GF, GA, GD, Point)
        matrix = data.iloc[:, 1:].values

        # Menghitung SVD
        U, sigma, VT = np.linalg.svd(matrix)

        sigma_matrix = np.zeros((U.shape[0], VT.shape[0]))
        np.fill_diagonal(sigma_matrix, sigma)
        sigma_matrix_formatted = np.array2string(sigma_matrix, formatter={'float_kind':lambda x: "%.3f" % x}, separator=', ')
        sigma_formatted = np.array2string(sigma, formatter={'float_kind':lambda x: "%.3f" % x}, separator=', ')
        VT_formatted = np.array2string(VT, formatter={'float_kind':lambda x: "%.3f" % x}, separator=', ')

        # Menampilkan hasil
        print("\nMatriks U tidak ditampilkan karena sangat panjang.")
        print("\nMatriks Sigma:")
        print(sigma_matrix_formatted)
        print("\nNilai Singular (Sigma):")
        print(sigma_formatted)
        print("\nMatriks V Transpose:")
        print(VT_formatted)

        # Menghitung dan menampilkan entropy
        entropy = calc_entropy(sigma)
        print("\nNilai Entropy:", entropy)

        # Menyimpan hasil ke file
        with open(output_path, "w") as f:
            f.write("Matriks U tidak ditampilkan karena sangat panjang.\n")
            f.write("\nMatriks Sigma:\n")
            f.write(sigma_matrix_formatted + "\n")
            f.write("\nNilai Singular (Sigma):\n")
            f.write(sigma_formatted + "\n")
            f.write("\nMatriks V Transpose:\n")
            f.write(VT_formatted + "\n")
            f.write("\nNilai Entropy:\n")
            f.write(str(entropy) + "\n")

    except FileNotFoundError:
        print("Error: File tidak ditemukan!")
    except Exception as e:
        print("Error:", str(e))