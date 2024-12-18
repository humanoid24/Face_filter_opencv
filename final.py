import cv2

# Muat classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

hat = cv2.imread('topi.png', -1)  # Gambar topi dengan transparansi
mask = cv2.imread('topeng.png', -1)  # Gambar topeng dengan transparansi
kacamata = cv2.imread('kacamata.png', -1)  # Gambar kacamata dengan transparansi
mask_tobi = cv2.imread('topeng_tobi.png', -1) # Gambar topeng
moustache = cv2.imread('kumis.png', -1)
wig = cv2.imread('wig.png', -1) # Gambar wig

filters = [hat, mask, kacamata, mask_tobi, moustache, wig]  # Daftar filter

current_filter = 0

# Variabel untuk menyimpan koordinat klik
click_coords = None

def add_filter(frame, face, filter_image, is_hat, is_glass, is_mask, is_maskTobi, is_moustache, is_wig):
    x, y, w, h = face  # Ambil koordinat wajah
    if is_hat:
        frame = put_hat(frame, filter_image, x, y, w, h)
    if is_glass:
        frame = put_glass(frame, filter_image, x, y, w, h)
    if is_mask:
        frame = put_mask(frame, filter_image, x, y, w, h)
    if is_maskTobi:
        frame = put_maskTobi(frame, filter_image, x, y, w, h)
    if is_moustache:
        frame = put_moustache(frame, filter_image, x, y, w, h)
    if is_wig:
        frame = put_wig(frame,filter_image, x, y, w, h)
    return frame

def put_wig(frame, wig, x, y, w, h):
    wig_width = int(1.5 * w)
    wig_height = int(1 * h)
    wig_resized = cv2.resize(wig, (wig_width, wig_height))

    offset_x = int(0.2 * wig_width)

    for i in range(wig_height):
        for j in range(wig_width):
            if wig_resized[i][j][3] != 0:  # Memeriksa alpha channel
                frame[y - int(0.5 * h) + i, x + j - offset_x] = wig_resized[i][j][:3]  # Hanya mengubah BGR channel
    return frame

def put_moustache(frame, moustache, x, y, w, h):
    moustache_width = w
    moustache_height = int(0.4 * h)  # Atur tinggi kumis sesuai proporsi wajah
    moustache_resized = cv2.resize(moustache, (moustache_width, moustache_height))  # Resize kumis

    for i in range(moustache_height):
        for j in range(moustache_width):
            if moustache_resized[i][j][3] != 0:  # Memeriksa alpha channel
                frame[y + int(0.55 * h) + i, x + j] = moustache_resized[i][j][:3]  # Hanya mengubah BGR channel
    return frame


def put_maskTobi(frame, mask_tobi, x, y, w, h):
    # Resize mask untuk menutupi seluruh wajah
    mask_width = int(w * 1.4)  # Tambah 20% lebar
    mask_height = int(h * 1.3)  # Tambah 20% tinggi
    mask_tobi = cv2.resize(mask_tobi, (mask_width, mask_height))

    # Mengatur offset untuk memindahkan topeng sedikit ke kiri
    offset_x = int(0.11 * mask_width)  # Misalnya, geser 10% dari lebar mask
    offset_y = int(0.10 * mask_height)  # Geser 10% dari tinggi mask

    # Tempatkan mask di posisi yang tepat dengan offset
    for i in range(mask_height):
        for j in range(mask_width):
            # Memeriksa apakah piksel pada mask memiliki transparansi
            if mask_tobi[i][j][3] != 0:  # Memeriksa alpha channel
                if (y + i - offset_y) < frame.shape[0] and (x + j - offset_x) < frame.shape[1]:
                    frame[y + i - offset_y, x + j - offset_x] = mask_tobi[i][j][:3]  # Hanya mengubah BGR channel

    return frame

def put_mask(frame, mask, x, y, w, h):
    # Resize mask untuk menutupi seluruh wajah
    mask_width = int(w * 1.4)  # Tambah 20% lebar
    mask_height = int(h * 1.3)  # Tambah 20% tinggi
    mask = cv2.resize(mask, (mask_width, mask_height))

    # Mengatur offset untuk memindahkan topeng sedikit ke kiri
    offset_x = int(0.10 * mask_width)  # Misalnya, geser 10% dari lebar mask
    offset_y = int(0.04 * mask_height)  # Geser 10% dari tinggi mask

    # Tempatkan mask di posisi yang tepat dengan offset
    for i in range(mask_height):
        for j in range(mask_width):
            # Memeriksa apakah piksel pada mask memiliki transparansi
            if mask[i][j][3] != 0:  # Memeriksa alpha channel
                if (y + i - offset_y) < frame.shape[0] and (x + j - offset_x) < frame.shape[1]:
                    frame[y + i - offset_y, x + j - offset_x] = mask[i][j][:3]  # Hanya mengubah BGR channel

    return frame

def put_hat(frame, hat, x, y, w, h):
    hat_width = w
    hat_height = int(0.6 * h)
    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            if hat[i][j][3] != 0:  # Memeriksa alpha channel
                frame[y - hat_height + i, x + j] = hat[i][j][:3]  # Hanya mengubah BGR channel
    return frame

def put_glass(frame, glass, x, y, w, h):
    glass_width = w
    glass_height = int(0.6 * h)
    glass = cv2.resize(glass, (glass_width, glass_height))

    for i in range(glass_height):
        for j in range(glass_width):
            if glass[i][j][3] != 0:  # Memeriksa alpha channel
                frame[y + int(0.15 * h) + i, x + j] = glass[i][j][:3]  # Hanya mengubah BGR channel
    return frame

# Fungsi callback untuk menangani klik mouse
def get_coordinates(event, x, y, flags, param):
    global click_coords
    if event == cv2.EVENT_LBUTTONDOWN:  # Cek jika klik kiri dilakukan
        click_coords = (x, y)  # Simpan koordinat klik
        print(f'Koordinat: (x={x}, y={y})')  # Menampilkan koordinat

cap = cv2.VideoCapture(0)

# Tampilkan jendela video dan daftarkan fungsi callback
cv2.namedWindow('Face Filter')
cv2.setMouseCallback('Face Filter', get_coordinates)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for face in faces:
        # Menambahkan filter
        is_hat = current_filter == 0  # Cek apakah filter yang aktif adalah topi
        is_mask = current_filter == 1  # Cek apakah filter yang aktif adalah topeng
        is_glass = current_filter == 2  # Cek apakah filter yang aktif adalah kacamata
        is_maskTobi = current_filter == 3
        is_moustache = current_filter == 4
        is_wig = current_filter == 5

        # Panggil fungsi untuk menambahkan filter
        frame = add_filter(frame, face, filters[current_filter], is_hat, is_glass, is_mask, is_maskTobi, is_moustache, is_wig)

        # Gambar kotak di sekitar wajah
        (x, y, w, h) = face
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Kotak biru di wajah
        # Tampilkan koordinat wajah
        #cv2.putText(frame, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Jika ada koordinat klik, gambar kotak
    # if click_coords:
    #    x, y = click_coords
    #    cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 2)  # Gambar kotak hijau

    cv2.imshow('Face Filter', frame)

    # Ganti filter dengan menekan tombol
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):  # Tekan 'n' untuk ganti filter
        current_filter = (current_filter + 1) % (len(filters))  # Menambahkan 1 untuk rambut dan kumis

cap.release()
cv2.destroyAllWindows()
