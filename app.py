import av
import io, os, time
import numpy as np
import requests
import threading
import queue, tempfile, openai

from collections import deque
from typing import List

import streamlit as st
from audiorecorder import audiorecorder
from pydub import AudioSegment
from streamlit_chat import message
from streamlit_option_menu import option_menu
from streamlit_webrtc import WebRtcMode, webrtc_streamer


openai.api_key = ""  # OpenAI API Key 입력

# 개인정보 제공 동의 문구


def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    """
    Save an audio segment to a .wav file.
    Args:
        audio_segment (AudioSegment): The audio segment to be saved.
        base_filename (str): The base filename to use for the saved .wav file.
    """
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")


def transcribe(audio_segment: AudioSegment, debug: bool = False) -> str:
    """
    Transcribe an audio segment using OpenAI's Whisper ASR system.
    Args:
        audio_segment (AudioSegment): The audio segment to transcribe.
        debug (bool): If True, save the audio segment for debugging purposes.
    Returns:
        str: The transcribed text.
    """
    if debug:
        save_audio(audio_segment, "debug_audio")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_segment.export(tmpfile.name, format="wav")
        answer = openai.Audio.transcribe(
            "whisper-1",
            tmpfile,
            temperature=0.2,
            prompt="",
        )["text"]
        tmpfile.close()
        os.remove(tmpfile.name)
        return answer


def frame_energy(frame):
    """
    Compute the energy of an audio frame.
    Args:
        frame (VideoTransformerBase.Frame): The audio frame to compute the energy of.
    Returns:
        float: The energy of the frame.
    """
    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
    return np.sqrt(np.mean(samples ** 2))


def process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold):
    """
    Process a list of audio frames.
    Args:
        audio_frames (list[VideoTransformerBase.Frame]): The list of audio frames to process.
        sound_chunk (AudioSegment): The current sound chunk.
        silence_frames (int): The current number of silence frames.
        energy_threshold (int): The energy threshold to use for silence detection.
    Returns:
        tuple[AudioSegment, int]: The updated sound chunk and number of silence frames.
    """
    for audio_frame in audio_frames:
        sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)

        energy = frame_energy(audio_frame)
        if energy < energy_threshold:
            silence_frames += 1
        else:
            silence_frames = 0

    return sound_chunk, silence_frames


def add_frame_to_chunk(audio_frame, sound_chunk):
    """
    Add an audio frame to a sound chunk.
    Args:
        audio_frame (VideoTransformerBase.Frame): The audio frame to add.
        sound_chunk (AudioSegment): The current sound chunk.
    Returns:
        AudioSegment: The updated sound chunk.
    """
    sound = AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    sound_chunk += sound
    return sound_chunk


def handle_silence(sound_chunk, silence_frames, silence_frames_threshold, text_output):
    """
    Handle silence in the audio stream.
    Args:
        sound_chunk (AudioSegment): The current sound chunk.
        silence_frames (int): The current number of silence frames.
        silence_frames_threshold (int): The silence frames threshold.
        text_output (st.empty): The Streamlit text output object.
    Returns:
        tuple[AudioSegment, int]: The updated sound chunk and number of silence frames.
    """
    if silence_frames >= silence_frames_threshold:
        if len(sound_chunk) > 0:
            text = transcribe(sound_chunk)
            text_output.write(text)
            sound_chunk = AudioSegment.empty()
            silence_frames = 0

    return sound_chunk, silence_frames


def handle_queue_empty(sound_chunk, text_output):
    """
    Handle the case where the audio frame queue is empty.
    Args:
        sound_chunk (AudioSegment): The current sound chunk.
        text_output (st.empty): The Streamlit text output object.
    Returns:
        AudioSegment: The updated sound chunk.
    """
    if len(sound_chunk) > 0:
        text = transcribe(sound_chunk)
        text_output.write(text)
        sound_chunk = AudioSegment.empty()

    return sound_chunk


def app_sst(
        status_indicator,
        text_output,
        timeout=3,
        energy_threshold=2000,
        silence_frames_threshold=100
):
    """
    The main application function for real-time speech-to-text.
    This function creates a WebRTC streamer, starts receiving audio data, processes the audio frames,
    and transcribes the audio into text when there is silence longer than a certain threshold.
    Args:
        status_indicator: A Streamlit object for showing the status (running or stopping).
        text_output: A Streamlit object for showing the transcribed text.
        timeout (int, optional): Timeout for getting frames from the audio receiver. Default is 3 seconds.
        energy_threshold (int, optional): The energy threshold below which a frame is considered silence. Default is 2000.
        silence_frames_threshold (int, optional): The number of consecutive silence frames to trigger transcription. Default is 100 frames.
    """
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )

    sound_chunk = AudioSegment.empty()
    silence_frames = 0

    while True:
        if webrtc_ctx.audio_receiver:
            status_indicator.write("Running. Say something!")

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                sound_chunk = handle_queue_empty(sound_chunk, text_output)
                continue

            sound_chunk, silence_frames = process_audio_frames(audio_frames, sound_chunk, silence_frames,
                                                               energy_threshold)
            sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, silence_frames_threshold,
                                                         text_output)
        else:
            status_indicator.write("Stopping.")
            if len(sound_chunk) > 0:
                text = transcribe(sound_chunk.raw_data)
                text_output.write(text)
            break

def app_sst_with_video(
        status_indicator,
        text_output,
        timeout=3,
        energy_threshold=2000,
        silence_frames_threshold=100
):

    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
            frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-with-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        audio_receiver_size=1024,
        media_stream_constraints={"video": True, "audio": True},
    )

    sound_chunk = AudioSegment.empty()
    silence_frames = 0

    while True:
        if webrtc_ctx.state.playing:
            status_indicator.write("Running. Say something!")

            try:
                audio_frames = []
                with frames_deque_lock:
                    while len(frames_deque) > 0:
                        frame = frames_deque.popleft()
                        audio_frames.append(frame)

                # audio_frames = webrtc_ctx.video_receiver.get_frames(timeout=timeout)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                sound_chunk = handle_queue_empty(sound_chunk, text_output)
                continue

            sound_chunk, silence_frames = process_audio_frames(audio_frames, sound_chunk, silence_frames,
                                                               energy_threshold)
            sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, silence_frames_threshold,
                                                         text_output)
        else:
            status_indicator.write("Stopping.")
            if len(sound_chunk) > 0:
                text = transcribe(sound_chunk.raw_data)
                text_output.write(text)
            break





# API 설정
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    # [{"role": "user", "content": "akdjflkadk"},
    # {"role": "assistant", "content": "e4ka;slkjf;lk"}]

host_url = "http://localhost:8000"
chat_url = f"{host_url}/chat"
transcribe_url = f"{host_url}/transcribe"

# 이미지 경로
image_home = './images/home/'
image_tutorial = './images/tutorial/'
image_developers = './images/developers/'

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Face & Talk", ["Home", 'Tutorial', 'Practice', 'chatBOT', 'developers'],
                           icons=['house', '1-circle', '2-circle', '3-circle', '4-circle'],
                           menu_icon="chat-heart",
                           styles={"nav-link-selected": {"background-color": "#D8B4F8"}},
                           default_index=0)

# 2. Home 페이지 #################################################################################
if selected == "Home":
    st.markdown("<h2 style='text-align: center; color: black; '>"
                "<span style='color:#7091F5'> FACE & TALK</span>"
                "</h2>",
                unsafe_allow_html=True)
    main_image = st.container()
    st.markdown("<h4 style='text-align: center; color: grey; '> "
                "안녕하세요.<br><br>"
                "발표 연습하세요.<br><br>"
                "이 홈페이지는 openAI 를 활용한 홈페이지 입니다. "
                "</h4>",
                unsafe_allow_html=True)

    with main_image:
        image_file = st.empty()
        for i in range(1, 13):
            image_file.image(image_home + f"{i}.png")
            time.sleep(0.2)
            

# 3. Tutorial 페이지 #################################################################################

if selected == "Tutorial":
    with st.container():
        st.markdown("<h2 style='text-align: center; color: black; font-size: 20;'>"
                    "AI 면접에 들어가기 앞서,<br>"
                    "기초 환경 세팅과 주의사항을 숙지해주세요"
                    " "
                    "</h2>",
                    unsafe_allow_html=True)

        # horizontal sidebar menu
        selected2 = option_menu(None, ["First", "Second", "Third"],
                                icons=['1-circle', '2-circle', '3-circle'],
                                menu_icon="cast",
                                styles ={"nav-link-selected": {"background-color": "#7091F5"}},
                                default_index=0,
                                orientation="horizontal")
        if selected2 == "First":
            # image load
            with st.columns(3)[1]:
                st.image(image_tutorial + "face_front.png")
            
            st.markdown("<p style='text-align: center; color: grey;'>"
                        "얼굴 전체가 화면에 들어오게 캠을 조정해주세요.<br>"
                        "너무 가깝거나 멀다면 얼굴 식별이 힘들 수 있습니다."
                        "</p>",
                        unsafe_allow_html=True)

        if selected2 == "Second":
            with st.columns(3)[1]:
                st.image(image_tutorial + "face_lateral.png")

            st.markdown("<p style='text-align: center; color: grey; font-size: 10;'>"
                        "얼굴 전체가 나오더라도 측면을 비추는 방향으로 캠을 두지 마십시오.<br>"
                        "면접이 진행되는 중에도 얼굴의 측면이 아닌 정면을 잘 비출 수 있도록 캠을 신경써주세요."
                        "</p>",
                        unsafe_allow_html=True)

        if selected2 == "Third":
            with st.columns(3)[1]:
                st.image(image_tutorial + "cassette.png", width=300)

            st.markdown("<p style='text-align: center; color: grey; font-size: 10;'>"
                        "질문에 대한 답변이 잘 녹화, 녹음될 수 있도록 유의하세요.<br>"
                        "잡음이 발생할 수 있는 주변 환경을 피하고 큰 목소리를 유지해주세요."
                        "</p>",
                        unsafe_allow_html=True)
            
# 4. Practice 페이지 #################################################################################

if selected == "Practice":
    def timer(container, limit_time: int):
        for seconds in range(limit_time):
            container.text(f"{limit_time - seconds} seconds left")
            time.sleep(1)

    tab1, tab2, tab3, tab4 = st.tabs(["녹화 테스트", "질문 생성기", "영상 + 대본", "대본"])
    with tab1:
        # 영상
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            # flipped = img[::-1,:,:] if flip else img
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        webrtc_streamer(key="example",
                        video_frame_callback=video_frame_callback)

    with tab2:
        generate_ad_slogan_url = "http://localhost:8000/create_ad_slogan"

        product_name = st.text_input("발표 종류")
        details = st.text_input("주요 내용")
        options = st.multiselect("질문 문구의 톤앤 매너", options=["기본", "과장스럽게", "차분한", "웃긴"], default=["기본"])

        if st.button("질문 생성"):
            try:
                response = requests.post(generate_ad_slogan_url,
                                         json={"product_name": product_name,
                                               "details": details,
                                               "tone_and_manner": ', '.join(options)})
                ad_slogan = response.json()['ad_slogan']
                st.success(ad_slogan)
            except:
                st.error("예상치 못한 에러가 발생했습니다")


    with tab3:
        st.title("Real-time Speech-to-Text")
        status_indicator = st.empty()
        text_output = st.container()
        app_sst_with_video(status_indicator, text_output)
            
    with tab4:
        st.title("Real-time Speech-to-Text")
        status_indicator = st.empty()
        text_output = st.empty()
        app_sst(status_indicator, text_output)

# 5. chatBOT 페이지 #################################################################################
if selected == "chatBOT":
    st.markdown("<h2 style='text-align: center; color: black; font-size: 10;'>"
                "ChatBOT SERVICE"
                "</h2>",
                unsafe_allow_html=True)
    
    def stt(audio_bytes):
        audio_file = io.BytesIO(audio_bytes)
        files = {"audio_file": ("audio.wav", audio_file, "audio/wav")}
        response = requests.post(transcribe_url, files=files)
        text = response.json()['text']
        return text


    def chat(text):
        user_turn = {"role": "user", "content": text}
        messages = st.session_state['messages']
        resp = requests.post(chat_url, json={"messages": messages + [user_turn]})
        assistant_turn = resp.json()

        st.session_state['messages'].append(user_turn)
        st.session_state['messages'].append(assistant_turn)

    row1 = st.container()
    row2 = st.container()

    with row2:
        audio = audiorecorder("Click to recoder", "Recording...")
        if len(audio) > 0:
            audio_bytes = audio.tobytes()
            st.audio(audio_bytes)

            text = stt(audio_bytes)
            chat(text)

    with row1:
        for i, msg_obj in enumerate(st.session_state['messages']):
            msg = msg_obj['content']

            is_user = False
            if i % 2 == 0:
                is_user = True

            message(msg, is_user=is_user, key=f"chat_{i}")


# 6. developers 페이지 #################################################################################
if selected == "developers":
    table = st.container()
    col1, col2 = st.columns(2)

    image_paths = [image_developers + "python.png",
                   image_developers + "streamlit.png",
                   image_developers + "fastapi.png",
                   image_developers + "whisper.png",
                   image_developers + "uvi.png"]
    # 이미지 크기 조정
    image_size = 50  # 원하는 이미지 크기 (너비, 높이)
    with table:
        with st.columns(3)[1]:
            st.image(image_developers + "last.jpg")

        with col1:
            st.markdown('<p style="text-align: left; color: black; font-weight: bold; font-size:30px;"> '
                        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-app" viewBox="0 0 16 16"> '
                        '<path d="M11 2a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3V5a3 3 0 0 1 3-3h6zM5 1a4 4 0 0 0-4 4v6a4 4 0 0 0 4 4h6a4 4 0 0 0 4-4V5a4 4 0 0 0-4-4H5z"/>'
                        '</svg> developers</p>'
                        
                        '<p style="text-align: left; color: black; font-size: 15px;">'
                        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-person-heart" viewBox="0 0 16 16">'
                        '<path d="M9 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm-9 8c0 1 1 1 1 1h10s1 0 1-1-1-4-6-4-6 3-6 4Zm13.5-8.09c1.387-1.425 4.855 1.07 0 4.277-4.854-3.207-1.387-5.702 0-4.276Z"/>'
                        '</svg> 팀장: 박은애</p>'
                        
                        '<p style="text-align: left; color: black; font-size: 15px;">'
                        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-person-heart" viewBox="0 0 16 16">'
                        '<path d="M9 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm-9 8c0 1 1 1 1 1h10s1 0 1-1-1-4-6-4-6 3-6 4Zm13.5-8.09c1.387-1.425 4.855 1.07 0 4.277-4.854-3.207-1.387-5.702 0-4.276Z"/>'
                        '</svg> 팀원: 류정인</p>'
                        
                        '<p style="text-align: left; color: black; font-size: 15px;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-person-heart" viewBox="0 0 16 16">'
                        '<path d="M9 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm-9 8c0 1 1 1 1 1h10s1 0 1-1-1-4-6-4-6 3-6 4Zm13.5-8.09c1.387-1.425 4.855 1.07 0 4.277-4.854-3.207-1.387-5.702 0-4.276Z"/>'
                        '</svg> 팀원: 홍세은</p>'
        
                        '<p style="text-align: left; color: black; font-size: 15px;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-person-heart" viewBox="0 0 16 16">'
                        '<path d="M9 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm-9 8c0 1 1 1 1 1h10s1 0 1-1-1-4-6-4-6 3-6 4Zm13.5-8.09c1.387-1.425 4.855 1.07 0 4.277-4.854-3.207-1.387-5.702 0-4.276Z"/>'
                        '</svg> 팀원: 공여진</p>',

                        unsafe_allow_html=True)

        with col2:
            st.markdown('<p style="text-align: right; color: black; font-weight: bold; font-size:15px;">'
                        '사용 기술 스택:  '
                        '</p>',
                        unsafe_allow_html=True)
            st.image(image_paths, width=image_size)
            st.markdown('<p style="text-align: right; color: black; font-weight: bold; font-size:15px;">'
                        'IMG 출처:Freepic'
                        '</p>',
                        unsafe_allow_html=True)
