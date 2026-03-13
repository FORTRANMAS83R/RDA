import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy import constants
from scipy.interpolate import interp1d
def load_sar_data(filename):
    """ Charge les données d'un fichier .mat. """
    try:
        mat_data = scipy.io.loadmat(filename)
        data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        return data
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' est introuvable.")
        return None
    except Exception as e:
        print(f"Une erreur s'est produite lors du chargement de '{filename}' : {e}")
        return None


class RDA:
    def __init__(self, fs_hz, T_chirp_s, B_hz, fc_hz):
        self.fs_hz = fs_hz
        self.T_chirp_s = T_chirp_s
        self.B_hz = B_hz
        self.fc_hz = fc_hz
        self.c_m_s = constants.speed_of_light
        self.symbol = None
        self.window = None

    def build_chirp(self, Td, window_r = True):
        Tp = self.T_chirp_s
        Kr = self.B_hz / Tp
        fs = self.fs_hz
        
        self.N_chirp = int(fs * Tp)
        offset = Td / 2 - Tp / 2
        t_centered = (np.arange(self.N_chirp) - self.N_chirp / 2) / fs
        tau = t_centered + offset
        
        
        phase = np.pi * Kr * (tau - (Td/2 - Tp/2))**2
        if window_r: 
            self.window = hann(self.N_chirp)
            self.symbol = self.window * np.exp(1j * phase)
        else: 
            self.symbol = np.exp(1j * phase)
        
        # plt.plot(np.real(self.symbol))
        # plt.show() # Commenté pour ne pas bloquer le pipeline automatique

    def range_compression(self, mat_raw):
        nf = mat_raw.shape[1]
        H = np.conj(np.fft.fft(self.symbol, n=nf))
        S = np.fft.fft(mat_raw, n=nf, axis=1)
        out_freq = S * H
        compressed_matrix = np.fft.ifftshift(np.fft.ifft(out_freq, axis=1), axes=1)
        return compressed_matrix
 
    def rcmc(self, N_slow, N_fast, data, dur, Ro, vp):
        # Ajout de 'vp' dans les arguments
        self.eta = np.linspace(-dur / 2, dur / 2, N_slow)
        delta_R = (vp**2 * self.eta**2) / (2 * Ro)
        delta_tau = 2 * delta_R / self.c_m_s
        
        f_tau = np.fft.fftfreq(N_fast, d=1/self.fs_hz)
        Src_f = np.fft.fft(data, axis=1)
        
        phase_shift = np.exp(1j * 2 * np.pi * f_tau[None, :] * delta_tau[:, None])
        Src_f_shifted = Src_f * phase_shift
        
        self.src_rcmc = np.fft.ifft(Src_f_shifted, axis=1)
        return self.src_rcmc

    def azimuth_compression(self, vp, Ro, window_az=True):
        # Ajout de 'Ro' dans les arguments
        Ka = (2 * vp**2) / ((self.c_m_s / self.fc_hz) * Ro)
        s_ref_a = np.exp(-1j * np.pi * Ka * self.eta**2)
        if window_az: 
            print('in')
            window_a = hann(len(self.eta))
            s_ref_a = s_ref_a * window_a  
        S_a = np.fft.fft(self.src_rcmc, axis=0)
        S_ref_a_f = np.fft.fft(s_ref_a)
        
        self.image_finale = np.fft.ifft(S_a * np.conj(S_ref_a_f)[:, None], axis=0)
        return np.fft.ifftshift(self.image_finale, axes=0)


def process_and_visualize(conf, dur, visu_range=False):
    """
    Exécute la chaîne RDA complète à partir d'un dictionnaire de configuration.
    """
    print(f"--- Début du traitement pour le fichier : {conf['file_path']} (Durée: {dur}s) ---")
    
    # 1. Extraction des variables de configuration
    file_path = conf['file_path']
    PRF = conf['PRF']
    vp = conf['vp']
    fc = conf['fc']
    Tp = conf['Tp']
    B0 = conf['B0']
    Ro = conf['Ro']
    fs = 2 * B0

    window_r = conf['window_r']
    window_az = conf['window_az']

    print(f'window_r: {window_r}, window_az:{window_az}')

    # 2. Chargement des données
    data_raw = load_sar_data(file_path)
    if data_raw is None:
        return
    data = np.array(data_raw['s'])
    N_slow, N_fast = data.shape
    Td = N_fast / fs

    # 3. Pipeline de traitement RDA
    r = RDA(fs, Tp, B0, fc)
    
    print("1/4 - Création du chirp...")
    r.build_chirp(Td, window_r=window_r)
    
    print("2/4 - Compression en distance...")
    compressed = r.range_compression(data)
    
    if visu_range: 
        print("\n--- Validation de la compression en distance ---")

        c = constants.speed_of_light
        
        # 1. Trouver le pixel de l'énergie maximale (le centre de la cible)
        idx_az, idx_rg = np.unravel_index(np.argmax(np.abs(compressed)), compressed.shape)
        
        # 2. Extraire la coupe en distance (1D) sur cette ligne d'azimut
        slice_rg = np.abs(compressed[idx_az, :])
        
        # 3. Créer un axe de distance en mètres
        dr = c / (2 * fs) # Taille d'un pixel en distance
        r_axis = np.arange(len(slice_rg)) * dr
        
        # 4. Interpolation cubique autour du pic pour une mesure de précision
        # On prend +/- 15 pixels autour du maximum
        w_size = 50 
        r_zoom = r_axis[idx_rg - w_size : idx_rg + w_size]
        slice_zoom = slice_rg[idx_rg - w_size : idx_rg + w_size]
        
        f_interp = interp1d(r_zoom, slice_zoom, kind='cubic')
        r_dense = np.linspace(r_zoom[0], r_zoom[-1], 2000) # Sur-échantillonnage
        slice_dense = f_interp(r_dense)
        
        # 5. Calcul de la largeur à -3dB
        peak_val = np.max(slice_dense)
        val_3db = peak_val / np.sqrt(2) # -3dB en amplitude correspond à Max / sqrt(2)
        
        indices_3db = np.where(slice_dense >= val_3db)[0]
        res_mesuree = r_dense[indices_3db[-1]] - r_dense[indices_3db[0]]
        
        # 6. Comparaison avec la théorie
        res_theorique = c / (2 * B0)
        
        print(f"Résolution théorique (c/2B) : {res_theorique:.3f} m")
        print(f"Résolution mesurée (-3dB)   : {res_mesuree:.3f} m")
        print(f"Erreur relative             : {abs(res_theorique - res_mesuree)/res_theorique * 100:.2f} %")

        # 7. Affichage de la coupe
        plt.figure(figsize=(8, 5))
        # On normalise par rapport au pic et on passe en dB
        plt.plot(r_dense, 20 * np.log10(slice_dense / peak_val), label='Sinc interpolé', color='blue')
        plt.plot(r_zoom, 20 * np.log10(slice_zoom / peak_val), 'o', label='Échantillons', color='red')
        
        plt.axhline(-3, color='k', linestyle='--', label='Niveau -3 dB')
        plt.axvline(r_dense[indices_3db[0]], color='green', linestyle=':', label=f'Largeur mesurée = {res_mesuree:.2f} m')
        plt.axvline(r_dense[indices_3db[-1]], color='green', linestyle=':')
        
        plt.title("Coupe en distance (Validation de la résolution)")
        plt.xlabel("Distance relative (m)")
        plt.ylabel("Amplitude (dB)")
        plt.ylim([-30, 2])
        plt.xlim([r_zoom[0], r_zoom[-1]])
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.show()
    
    print("3/4 - RCMC (Correction des migrations)...")
    rcmc_res = r.rcmc(N_slow, N_fast, compressed, dur, Ro, vp)
    
    print("4/4 - Compression en azimut...")
    img = r.azimuth_compression(vp, Ro, window_az=window_az)

    # 4. Visualisation
    print("Affichage des résultats...")
    plt.figure(figsize=(12, 10), constrained_layout=True)
    
    # -- Raw Data --
    plt.subplot(2, 2, 1)
    plt.title("Raw Data")
    plt.imshow(np.abs(data), aspect='auto', cmap='viridis')
    plt.ylabel("Azimuth (samples)")
    plt.xlabel("Range (samples)")

    # -- Range Compressed --
    plt.subplot(2, 2, 2)
    plt.title("Range Compressed")
    plt.imshow(np.abs(compressed), aspect='auto', cmap='viridis')
    plt.ylabel("Azimuth (samples)")
    plt.xlabel("Range (samples)")

    # -- RCMC --
    plt.subplot(2, 2, 3)
    plt.title("RCMC")
    plt.imshow(np.abs(rcmc_res), aspect='auto', cmap='viridis')
    plt.ylabel("Azimuth (samples)")
    plt.xlabel("Range (samples)")

    # -- Azimuth Compressed (en dB pour voir la cible) --
    plt.subplot(2, 2, 4)
    plt.title("Azimuth Compressed (dB)")
    img_db = 20 * np.log10(np.abs(img) + 1e-10)
    vmax = np.max(img_db)
    vmin = vmax - 40 # Dynamique de 40 dB
    plt.imshow(img_db, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    plt.ylabel("Azimuth (samples)")
    plt.xlabel("Range (samples)")

    plt.show()


if __name__ == "__main__":
    # Dictionnaire de configuration
    config = {
        "file_path": "onepointtarget_3s.mat",
        "PRF": 300,
        "vp": 200,
        "fc": 4.5e9,
        "Tp": 0.25e-5,
        "B0": 100e6,
        "theta": 45,
        "Ro": 20e3, 
        "window_r": True, 
        "window_az": False
    }
    
    # Appel de la méthode avec la configuration et la durée
    process_and_visualize(conf=config, dur=3, visu_range=True)

    config["file_path"] = "cpxtarget_3s.mat"
    process_and_visualize(conf=config, dur=3)

    config["file_path"] = "onepointtarget_6s.mat"
    process_and_visualize(conf=config, dur=6)

    config["file_path"] = "onepointtarget_noise_6s.mat"
    process_and_visualize(conf=config, dur=6)

    config["file_path"] = "cpxtarget_6s.mat"
    process_and_visualize(conf=config, dur=6)

