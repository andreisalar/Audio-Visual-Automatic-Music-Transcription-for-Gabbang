import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import find_peaks
import math
import time
import cv2
from sklearn.cluster import DBSCAN
from music21 import stream, note, chord, musicxml
from music21 import stream, note, tempo, midi, converter, environment, chord

# Record the start time
start_time = time.time()


# ==========================================AUDIO CODE==========================================

def onset_detection(audio_file):
    onset_detection_time_start = time.time()

    y, sr = librosa.load(audio_file)
    hop_length = 512
    n_fft = 2048
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Normalize the spectrogram
    spectrogram_normalized = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Detect onset times using spectral flux
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=spectral_flux, sr=sr, hop_length=hop_length)

    #converting frames to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    len_onset = len(onset_times)

    onset_frames_new = [int(x*30) for x in onset_times] # Convert seconds to FPS (30FPS)

    onset_detection_time_end = time.time()
    onset_detection_time_elapsed = onset_detection_time_end - onset_detection_time_start

    return len_onset, onset_times, onset_frames, spectrogram, spectrogram_normalized, sr, onset_frames_new, onset_detection_time_elapsed

def spectrogram_plot(audio_file, spectrogram_normalized):
    y, sr = librosa.load(audio_file)
    hop_length = 512
   
    # Plot the original audio waveform
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Original Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot the normalized spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(spectrogram_normalized, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


def pitch_detection(audio_file, len_onset, onset_times, onset_frames, spectrogram, sr):
    pitch_detection_time_start = time.time()

    hop_length = 512
    n_fft = 2048

    onset_pitch = []
    detected_pitches = []

    for idx, onsets in enumerate(onset_times):                                         #using frames to identify pitches
        
        # Convert onset time to frame index
        onset_frame = int(onsets * sr / hop_length)

        # Extract the frequency spectrum at the onset time
        spectrum = spectrogram[:, onset_frame]
        
        # Find the indices of the top k peaks in the spectrum. Get the value of k using chromagram

# --------------CHROMAGRAM CALCULATIONS FOR NUM_PITCHES---------------------
        y, sr1 = librosa.load(audio_file)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr1)
        # Get the start and end frame indices for the current onset

        start_frame = onset_frames[idx]

        if idx < (len_onset-1):
            end_frame = onset_frames[idx+1]
        else:
            end_frame = chromagram.shape[1]  # Use the total number of frames in the chromagram

        # Extract the chromagram segment within the onset
        chromagram_segment = chromagram[:, start_frame:end_frame]
        # Count the number of distinct pitches within the onset segment
        k = len(set(chromagram_segment.argmax(axis=0)))
        
        
        # ------------------end of chromagram code -----------------

        peak_indices = np.argpartition(spectrum, -k)[-k:]

        # Retrieve the pitch frequencies corresponding to the peak indices
        pitch_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[peak_indices]

        detected_pitches.append(pitch_frequencies)
        # Add the estimated pitch frequency to the list
        onset_pitch.append([onsets, pitch_frequencies])

    pitch_detection_time_end = time.time()
    pitch_detection_time_elapsed = pitch_detection_time_end - pitch_detection_time_start

    return detected_pitches, onset_pitch, chromagram, pitch_detection_time_elapsed
    

def calculate_note_frequency(frequency):
    
    # Map semitone offsets from C to note names
    note_names = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    # Check if the frequency value is valid
    if frequency <= 0:
        return "Invalid", "Invalid"

    # Calculate the number of semitones away from C
    n = 12 * math.log2(frequency / 261.63)  # Assuming reference frequency of C4 is 261.63 Hz

    # Calculate the note and octave
    semitones = round(n)
    octave = (semitones // 12) + 4
    note = note_names[semitones % 12]

    return note, octave

def text_file_generate(onset_times, detected_pitches):
    note_assignment_time_start = time.perf_counter()
    # OUTPUT OF NOTES AND PITCH TEXT FILE
    OnsetPitchFile = 'GABBANG1_100BPM_Onset_Pitch.txt'

    # Perform the for loop and generate the output
    onsetpitch = []
    for i in range(len(onset_times)):
        onset = onset_times[i]
        pitches = detected_pitches[i]
        onsetpitch.append([round(onset,4),sorted(pitches)])

    # Save the output to the text file
    with open(OnsetPitchFile, 'w') as file:
        for output_value in onsetpitch:
            file.write(str(output_value) + '\n')

    # -----------------------------------------

    # OUPUT TEXT FILE OF NOTES
    NoteFile = 'GABBANG1_100BPM_Notes.txt'


    notes_tracked = []

    # Iterate through the original matrix
    for row in onsetpitch:
        notes_row = []
        notes_row.append(row[0])  # Keep the first element as it is

        if isinstance(row[1], list):
            # If the second element is a list, convert its values to notes
            notes_sublist = []
            for element in row[1]:
                note, _= calculate_note_frequency(element)  # Ignore the octave value
                notes_sublist.append(note)
                notes_sublist = list(set(notes_sublist))
            notes_row.append(notes_sublist)
        else:
            # If the second element is not a list, keep its original value
            notes_row.append(row[1])

        notes_tracked.append(notes_row)

    tracked_notes = []
    # Print the new notes matrix
    for row in notes_tracked:
        tracked_notes.append(row)

    # Save the output to the text file
    with open(NoteFile, 'w') as file:
        for output_value in tracked_notes:
            file.write(str(output_value) + '\n') 

    note_assignment_time_end = time.perf_counter()
    note_assignment_time_elapsed = note_assignment_time_end - note_assignment_time_start
    # notes_only = [item[1] for item in tracked_notes]    #NOTES ONLY MATRIX
    return tracked_notes, note_assignment_time_elapsed


# ==========================MAIN CODE===============================

print("Kindly input the tuning of your Gabbang. The first bar shall be the right-most bar in the video")
firstnote = input("Note of the first bar: ")
secondnote = input("Note of the second bar: ")
thirdnote = input("Note of the third bar: ")
fourthnote = input("Note of the fourth bar: ")
fifthnote = input("Note of the fifth bar: ")


audio_file = "/home/andreisalar/Thesis/NEWEST AUDIO 0623/Track 1- 100bpm.wav"

len_onset, onset_times, onset_frames, spectrogram, spectrogram_normalized, sr, onset_frames_new, onset_detection_time_elapsed = onset_detection(audio_file)
plot = spectrogram_plot(audio_file, spectrogram_normalized)
detected_pitches, onset_pitch, chromagram, pitch_detection_time_elapsed = pitch_detection(audio_file, len_onset, onset_times, onset_frames, spectrogram, sr)
tracked_notes, note_assignment_time_elapsed = text_file_generate(onset_times, detected_pitches)   #TO BE ACCESSED WHEN COMPARING THE AUDIO AND VIDEO NOTES


# ==========================================VIDEO CODE==========================================

#function for line dection including brightness adjustment 

# Load the video
cap = cv2.VideoCapture("/home/andreisalar/Thesis/CROPPED/Gabbang 1 (100BPM).mp4")      #change the file name

# Read a frame from the video
ret, frame1 = cap.read()

# Resize the image for faster processing
frame1 = cv2.resize(frame1, (640, 480), interpolation=cv2.INTER_LINEAR)

#declaration of functions 

line_clustering_time_start = time.perf_counter()

#FUNCTION 1
# Perform line clustering post-processing
def line_clustering(lines, threshold_distance):
    # Create a similarity matrix
    similarity_matrix = np.zeros((len(lines), len(lines)))

    # Calculate similarity metric (distance) for each pair of lines
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i != j:
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                distance = np.abs(x1 - x3) + np.abs(y1 - y3) + np.abs(x2 - x4) + np.abs(y2 - y4)
                similarity_matrix[i][j] = distance

    # Apply DBSCAN clustering algorithm
    clustering = DBSCAN(eps=threshold_distance, min_samples=2).fit(similarity_matrix)
    labels = clustering.labels_

    return labels

# Declaration of variables with fixed values

    # Define the brightness adjustment factor
brightness_factor = 5.54

    # Define the clustering threshold distance
threshold_distance = 125

    # Define the minimum and maximum line lengths to consider
minimum_line_length = 235
maximum_line_length = 250

    # Define the distance threshold for averaging clusters
distance_threshold = 20

    # Define the coordinates of the ROI to extract
roi_x, roi_y, roi_w, roi_h = 240, 100, 180, 250


    # Define the lower and upper boundaries of the red color in HSV color space
red_lower = np.array([0, 120, 70])
red_upper = np.array([10, 255, 255])

    # Increment the frame counter
frame_count= 0 

    # Create a blank list for formatted lines 
formatted_lines = []

    #blank list of the center of the circle
circle_center = []

#Using the Onset Frames from Audio
onsets = onset_frames_new

i = 0


# Adjust the brightness of the frame
frame2 = cv2.convertScaleAbs(frame1, alpha=brightness_factor, beta=0)

# Extract the ROI from the frame
roi1 = frame1[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]


# Extract the ROI from the frame with adjusted brightness
roi2 = frame2[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

line_clustering_time_end = time.perf_counter()
line_clustering_time_elapsed = line_clustering_time_end - line_clustering_time_start


#====================Line detection for frame ====================

line_detection_time_start = time.time()

    # Convert the ROI to grayscale
gray_roi = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection to the ROI
sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.magnitude(sobel_x, sobel_y)
edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Enhance edges using morphological operations
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)
edges = cv2.erode(edges, kernel, iterations=2)

    # Apply Hough Line Transform to detect lines in the ROI using solid contours
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=180, minLineLength=200, maxLineGap=10)

    # Iterate over each line in the matrix
for line in lines:
    x1, y1, x2, y2 = line[0]  # Extract the four values
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if minimum_line_length <= line_length <= maximum_line_length:
        formatted_lines.append((x1, y1, x2, y2))  # Store the line coordinates as a tuple

    # Perform line clustering
labels = line_clustering(formatted_lines, threshold_distance)

    # Assign unique colors to the clustered lines
colors = np.random.randint(0, 255, size=(len(np.unique(labels)), 3), dtype=np.uint8)

    # Average the lines in each cluster within the distance threshold
averaged_lines = []
unique_labels = np.unique(labels)
for label in unique_labels:
    cluster_lines = [line for line, lbl in zip(formatted_lines, labels) if lbl == label]
    avg_line = np.mean(cluster_lines, axis=0, dtype=np.int32)
    averaged_lines.append(avg_line)

    # Filter the averaged lines based on the distance threshold
filtered_lines = []
x_axis = []
for line in averaged_lines:
    if len(filtered_lines) == 0:
        filtered_lines.append(line)
    else:
        x1, y1, x2, y2 = line
        distances = [np.sqrt((x2 - l[0]) ** 2 + (y2 - l[1]) ** 2) for l in filtered_lines]
        if np.min(distances) > distance_threshold:
            filtered_lines.append(line)
            
for line in filtered_lines:
    x1, y1, x2, y2 = line
    x_axis.append(x1)
    
sorted = sorted(x_axis)  

line_detection_time_end = time.time()
line_detection_time_elapsed = line_detection_time_end - line_detection_time_start

#====================color detection for the beaters ====================

color_detection_time_start = time.time()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Increment the frame counter
    frame_count += 1

    stop_frame = onsets[int(i)]

    # Resize the image for faster processing
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Extract the ROI from the frame
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

     # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the red color using the lower and upper boundaries
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles on the detected red color
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 5)
            cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 10)
            #print(x)            #print(y)
            
            # Use the x, y coordinates here to get the center of the mallets
            if frame_count == stop_frame:
                circle_center.append((x-240))
                if len(onsets) == i+1:
                    break
                i += 0.5
                
    
    # Draw the formatted lines on the video itself 
    for line, color in zip(filtered_lines, colors):
        x1, y1, x2, y2 = line
        cv2.line(roi, (x1, y1), (x2, y2), tuple(color.tolist()), 2)


    # Display the ROI
    cv2.imshow('frame', roi)

    
    # Exit if the 'q' key is pressed or the end of the video is reached
    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

color_detection_time_end = time.time()
color_detection_time_elapsed = color_detection_time_end - color_detection_time_start

#comparison of lines and center
barnumber_time_start = time.time()

bar = []
for a in circle_center:
    if a  < sorted[0]:
        bar.append(fifthnote)
    elif sorted[1] > a > sorted[0]:
        bar.append(fourthnote)
    elif sorted[2] > a > sorted [1]:
        bar.append(thirdnote)
    elif sorted[3] > a > sorted[2]:
        bar.append(secondnote)
    else:
        bar.append(firstnote)
mallet1 = bar[::2] #MAIN VIDEO OUTPUT
mallet2 = bar[1::2] #MAIN VIDEO OUTPUT

cap.release()
cv2.destroyAllWindows()


X = len(mallet1)
Y = len(mallet2)

Z = X-Y

if X != Y:
    for i in range(Z):
        mallet2.append(firstnote)

barnumber_time_end = time.time()
barnumber_time_elapsed = barnumber_time_end - barnumber_time_start


#======================EXTRACTING THE MALLET OUTPUTS INTO A TEXT FILE =======================

#======================END OF EXTRACTING THE MALLET OUTPUTS INTO A TEXT FILE =======================


# ====================== Translation of notes to numbers ======================

tabulation = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

for i in tracked_notes:
    notes = i[1]
    mapped_values = [tabulation[value] for value in notes]
    i[1] = mapped_values

# print(notes_tracked)    #it checks out


mapped_values = [tabulation[value] for value in mallet1]
mallet1 = mapped_values

# print(mallet1)  #it checks out

mapped_values = [tabulation[value] for value in mallet2]
mallet2 = mapped_values
# print(mallet2) #it checks out


# ====================== Translation of notes to numbers : DONE ======================



# ====================== Audio-Visual Comparison ======================
fusion_time_start = time.time()

final_notes=[]
gabbang_notes=[]
m= len(mallet1)   #to be used in final notes notation
checker = []      #to be used in accessing the checked_notes

for idx,i in enumerate(tracked_notes):   #ACCESSING THE ELEMENT IN NOTES TRACKED
    checked_notes = []

    for j in i[1:]:   #ACCESSING THE SECOND ELEMENT PER ELEMENTS IN NOTES TRACKED
        V1 = mallet1[idx]  #getting the value per index of mallet 1
        V2 = mallet2[idx]  #getting the value per index of mallet 2
        for k in j:
            if (k - V1) == 0 or (k - V2) == 0: 
                checked_notes.append(k)
            elif (k - V1) == 1 or (k - V1) == -1 :
                checked_notes.append(V1)
            elif (k - V2) == 1 or (k - V2) == -1:
                checked_notes.append(V2)
    checker.append(checked_notes)

for index, M in enumerate(checker):        #if there are no matching audio and video data, used the audio data
    if not M:
        checker[index] = tracked_notes[index][1]

# FINALS NOTES NOTATION
for i in range(m-1):
    final_notes.append([tracked_notes[i][0],list(set(checker[i]))])


# ==== Translation of numbers to notes =====
tabulation2 = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}

for i in final_notes:
    notes = i[1]
    mapped_values = [tabulation2[value] for value in notes]
    i[1] = mapped_values

# ==== End of translation of numbers to notes =====

         
with open('GABBANG1_100BPM_FUSED_NOTES.txt', 'w') as file:
    for timez in final_notes:
        file.write(str(timez) + '\n')

# print(final_notes)


# ==== Translation of notes to number of bars =====
tabulation3 = {
    firstnote : 1 , secondnote : 2 , thirdnote : 3 , fourthnote: 4 , fifthnote : 5      #WILL CHANGE DEPENDING ON THE CODE OF VIDEO
}

for i in final_notes:
    notes = i[1]
    bar_list = [[elem[0], [tabulation3.get(note) for note in elem[1] if note in tabulation3]] for elem in final_notes]
    bar_list = [[elem[0], [note for note in elem[1] if note is not None]] for elem in bar_list]


# ==== End of translation of notes to number of bars =====

with open('GABBANG1_100BPM_FUSED_NOTES_BARS.txt', 'w') as file:
    for timez in bar_list:
        file.write(str(timez) + '\n')

fusion_time_end = time.time()
fusion_time_elapsed = fusion_time_end - fusion_time_start

# ====================== End of Audio-Visual Comparison ======================

# ====================== SCORE PROCESSING

score_time_start = time.time()

sublists = [sublist[1] for sublist in bar_list]

def create_polyphonic_score(sublists):
    polyphonic_score = stream.Score()
    part = stream.Part()
    voice = stream.Voice()

    for sublist in sublists:
        if len(sublist) > 1:
            chord_notes = []
            chord_labels = []

            for note_value in sublist:
                if note_value == 1:
                    note_name = 'E4'
                    label = '1'
                elif note_value == 2:
                    note_name = 'G4'
                    label = '2'
                elif note_value == 3:
                    note_name = 'B4'
                    label = '3'
                elif note_value == 4:
                    note_name = 'D5'
                    label = '4'
                elif note_value == 5:
                    note_name = 'F5'
                    label = '5'
                else:
                    note_name = 'C4'  # Default note value if an invalid value is encountered
                    label = 'X'

                n = note.Note(note_name)
                n.duration.type = '16th'  # Set the duration to sixteenth note
                chord_notes.append(n)
                chord_labels.append(label)

            c = chord.Chord(chord_notes)
            c.addLyric(','.join(chord_labels))
            voice.append(c)
        else:
            note_value = sublist[0]

            if note_value == 1:
                note_name = 'E4'
                label = '1'
            elif note_value == 2:
                note_name = 'G4'
                label = '2'
            elif note_value == 3:
                note_name = 'B4'
                label = '3'
            elif note_value == 4:
                note_name = 'D5'
                label = '4'
            elif note_value == 5:
                note_name = 'F5'
                label = '5'
            else:
                note_name = 'C4'  # Default note value if an invalid value is encountered
                label = 'X'

            n = note.Note(note_name)
            n.duration.type = '16th'  # Set the duration to sixteenth note
            n.addLyric(label)
            voice.append(n)

    part.append(voice)
    polyphonic_score.insert(0, part)  # Insert the part at the beginning to maintain the correct order

    return polyphonic_score

# Example usage
score = create_polyphonic_score(sublists)
exporter = musicxml.m21ToXml.GeneralObjectExporter(score)
musicxml_output = exporter.parse()

# Save the polyphonic score as a MusicXML file
output_filename = 'polyphonic_score.xml'
with open(output_filename, 'w') as file:
    file.write(musicxml_output.decode())

print(f"Polyphonic score saved as '{output_filename}'.")

score_time_end = time.time()
score_time_elapsed = score_time_end - score_time_start

# ============== MIDI PROCESSING ==========================

midi_time_start = time.time()

gabbang_notes  = [sublist[1] for sublist in final_notes]
onset_notes = [sublist[0] for sublist in final_notes]

def create_midi_file(notes, onset_times, tempo_bpm = 120):
    # Create a stream object to store the notes
    midi_stream = stream.Stream()

    # Add the notes with their onset times to the stream
    for note_list, onset_time in zip(gabbang_notes, onset_notes):
        chord_notes = []

        # Create note objects for each note in the sublist
        for note_name in note_list:
            n = note.Note(note_name)
            chord_notes.append(n)

        # Create a chord object from the note objects
        c = chord.Chord(chord_notes)
        c.offset = onset_time
        midi_stream.append(c)

    # Set the tempo of the MIDI file
    tempo_marking = tempo.MetronomeMark(number=tempo_bpm)
    midi_stream.insert(0, tempo_marking)

    # Create a MIDI file object
    midi_file = midi.translate.streamToMidiFile(midi_stream)

    return midi_file


# Example usage
midi_file = create_midi_file(notes, onset_times, tempo_bpm=150)

# Save the MIDI file
output_filename = 'GABBANG1_100BPM_midi.mid'
midi_file.open(output_filename, 'wb')
midi_file.write()
midi_file.close()

print(f"MIDI file saved as '{output_filename}'.")

midi_time_end = time.time()
midi_time_elapsed = midi_time_end - midi_time_start

#===========RUNTIMES============
# print(f"Onset Detection elapsed time: {onset_detection_time_elapsed} seconds")
# print(f"Pitch Detection elapsed time: {pitch_detection_time_elapsed} seconds")
# print(f"Note Assignment elapsed time: {note_assignment_time_elapsed: .10f} seconds")
# print(f"Line Clustering elapsed time: {line_clustering_time_elapsed} seconds")
# print(f"Line Detection elapsed time: {line_detection_time_elapsed: .10f} seconds")
# print(f"Color Detection elapsed time: {color_detection_time_elapsed} seconds")
# print(f"Bar Number Assignment elapsed time: {barnumber_time_elapsed} seconds")
# print(f"Fusion elapsed time: {fusion_time_elapsed} seconds")
# print(f"Score Creation elapsed time: {score_time_elapsed} seconds")
# print(f"Midi Creation elapsed time: {midi_time_elapsed} seconds")

# Record the end time
end_time = time.time()

# Calculate the runtime
runtime = end_time - start_time

# Print the runtime
print("Overall System Runtime:", runtime, "seconds")

# Write the elapsed time data to a text file
filename = "AMT_Runtimes.txt"  # Specify the file name
with open(filename, "a") as file:
    file.write(f"Onset Detection elapsed time: {onset_detection_time_elapsed} seconds\n")
    file.write(f"Pitch Detection elapsed time: {pitch_detection_time_elapsed} seconds\n")
    file.write(f"Note Assignment elapsed time: {note_assignment_time_elapsed: .10f} seconds\n")
    file.write(f"Line Clustering elapsed time: {line_clustering_time_elapsed} seconds\n")
    file.write(f"Line Detection elapsed time: {line_detection_time_elapsed: .10f} seconds\n")
    file.write(f"Color Detection elapsed time: {color_detection_time_elapsed} seconds\n")
    file.write(f"Bar Number Assignment elapsed time: {barnumber_time_elapsed} seconds\n")
    file.write(f"Fusion elapsed time: {fusion_time_elapsed} seconds\n")
    file.write(f"Score Creation elapsed time: {score_time_elapsed} seconds\n")
    file.write(f"Midi Creation elapsed time: {midi_time_elapsed} seconds\n")
    file.write(f"Total elapsed time: {runtime} seconds\n")

