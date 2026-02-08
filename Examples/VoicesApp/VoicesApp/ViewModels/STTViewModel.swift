import Foundation
import SwiftUI
import MLXAudioSTT
import MLXAudioCore
import MLX
import AVFoundation
import Combine

@MainActor
@Observable
class STTViewModel {
    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var transcriptionText: String = ""
    var tokensPerSecond: Double = 0
    var peakMemory: Double = 0

    // Generation parameters
    var maxTokens: Int = 8192
    var temperature: Float = 0.0
    var language: String = "English"
    var chunkDuration: Float = 250.0

    // Model configuration
    var modelId: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    private var loadedModelId: String?

    // Audio file
    var selectedAudioURL: URL?
    var audioFileName: String?

    // Audio player state
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    // Recording state
    var isRecording: Bool { recorder.isRecording }
    var recordingDuration: TimeInterval { recorder.recordingDuration }
    var audioLevel: Float { recorder.audioLevel }

    private var model: Qwen3ASRModel?
    private let audioPlayer = AudioPlayerManager()
    private let recorder = AudioRecorderManager()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?

    var isModelLoaded: Bool {
        model != nil
    }

    init() {
        setupAudioPlayerObservers()
    }

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.isPlaying = value
            }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.currentTime = value
            }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.duration = value
            }
            .store(in: &cancellables)
    }

    func loadModel() async {
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            model = try await Qwen3ASRModel.fromPretrained(modelId)
            loadedModelId = modelId
            generationProgress = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false
    }

    func reloadModel() async {
        model = nil
        loadedModelId = nil
        Memory.clearCache()
        await loadModel()
    }

    func selectAudioFile(_ url: URL) {
        selectedAudioURL = url
        audioFileName = url.lastPathComponent
        audioPlayer.loadAudio(from: url)
    }

    func startTranscription() {
        guard let audioURL = selectedAudioURL else {
            errorMessage = "No audio file selected"
            return
        }

        generationTask = Task {
            await transcribe(audioURL: audioURL)
        }
    }

    func transcribe(audioURL: URL) async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        errorMessage = nil
        transcriptionText = ""
        generationProgress = "Loading audio..."
        tokensPerSecond = 0
        peakMemory = 0

        do {
            let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
            let targetRate = model.sampleRate

            let resampled: MLXArray
            if sampleRate != targetRate {
                generationProgress = "Resampling \(sampleRate)Hz → \(targetRate)Hz..."
                resampled = try resampleAudio(audioData, from: sampleRate, to: targetRate)
            } else {
                resampled = audioData
            }

            generationProgress = "Transcribing..."

            var tokenCount = 0
            for try await event in model.generateStream(
                audio: resampled,
                maxTokens: maxTokens,
                temperature: temperature,
                language: language,
                chunkDuration: chunkDuration
            ) {
                try Task.checkCancellation()

                switch event {
                case .token(let token):
                    transcriptionText += token
                    tokenCount += 1
                    generationProgress = "Transcribing... \(tokenCount) tokens"
                case .info(let info):
                    tokensPerSecond = info.tokensPerSecond
                    peakMemory = info.peakMemoryUsage
                case .result:
                    generationProgress = ""
                }
            }

            generationProgress = ""
        } catch is CancellationError {
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
    }

    // MARK: - Live Recording & Transcription
    //
    // Dual-chunk system:
    //  - Fast chunks (~2s): quick transcription for immediate feedback → pendingText
    //  - Correction chunks (~10s): re-transcribe full pending window → replaces pendingText, promotes to confirmedText
    //  - Display = confirmedText + pendingText

    private let fastInterval: Double = 2.0
    private let correctionInterval: Double = 10.0
    private let sampleRate = 16000

    private var liveTask: Task<Void, Never>?
    private var correctionTask: Task<Void, Never>?

    /// Text confirmed by correction chunks (high quality)
    private var confirmedText: String = ""
    /// Text from fast chunks (may have errors, will be replaced by correction)
    private var pendingText: String = ""
    /// Sample position where confirmedText ends
    private var confirmedSampleEnd: Int = 0
    /// Sample position where last fast chunk ended
    private var fastChunkEnd: Int = 0

    func startRecording() {
        errorMessage = nil
        transcriptionText = ""
        confirmedText = ""
        pendingText = ""
        confirmedSampleEnd = 0
        fastChunkEnd = 0
        tokensPerSecond = 0
        peakMemory = 0

        do {
            try recorder.startRecording()
        } catch {
            errorMessage = error.localizedDescription
            return
        }

        liveTask = Task {
            await liveTranscriptionLoop()
        }
    }

    private func liveTranscriptionLoop() async {
        var timeSinceCorrection: Double = 0

        while !Task.isCancelled && recorder.isRecording {
            try? await Task.sleep(for: .seconds(fastInterval))
            guard !Task.isCancelled && recorder.isRecording else { break }
            timeSinceCorrection += fastInterval

            // Fire off correction in the background if enough time has passed
            if timeSinceCorrection >= correctionInterval {
                timeSinceCorrection = 0
                startCorrectionChunk()
            }

            // Always run fast chunks (even while correction runs in background)
            await runFastChunk()
        }
    }

    /// Fast chunk: transcribe only new audio since last fast chunk. Streams tokens to pendingText.
    private func runFastChunk() async {
        guard let model = model else { return }
        guard let (audio, endPos) = recorder.getAudio(from: fastChunkEnd) else { return }
        fastChunkEnd = endPos

        isGenerating = true
        generationProgress = "Transcribing..."

        do {
            for try await event in model.generateStream(
                audio: audio,
                maxTokens: maxTokens,
                temperature: temperature,
                language: language,
                chunkDuration: chunkDuration
            ) {
                try Task.checkCancellation()
                switch event {
                case .token(let token):
                    pendingText += token
                    transcriptionText = confirmedText + pendingText
                case .info(let info):
                    tokensPerSecond = info.tokensPerSecond
                    peakMemory = info.peakMemoryUsage
                case .result:
                    break
                }
            }
        } catch is CancellationError {
            Memory.clearCache()
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
        }

        generationProgress = ""
        isGenerating = false
    }

    /// Fire-and-forget correction: re-transcribe all audio from confirmedSampleEnd in the background.
    /// When done, promotes result to confirmedText and resets pending. Skips if already running.
    private func startCorrectionChunk() {
        guard correctionTask == nil else { return }
        guard let model = model else { return }
        let currentEnd = recorder.sampleCount
        guard currentEnd > confirmedSampleEnd else { return }
        guard let (audio, endPos) = recorder.getAudio(from: confirmedSampleEnd) else { return }

        correctionTask = Task {
            var correctionText = ""
            do {
                for try await event in model.generateStream(
                    audio: audio,
                    maxTokens: maxTokens,
                    temperature: temperature,
                    language: language,
                    chunkDuration: chunkDuration
                ) {
                    try Task.checkCancellation()
                    switch event {
                    case .token(let token):
                        correctionText += token
                    case .info(let info):
                        tokensPerSecond = info.tokensPerSecond
                        peakMemory = info.peakMemoryUsage
                    case .result:
                        break
                    }
                }
            } catch is CancellationError {
                Memory.clearCache()
            } catch {
                errorMessage = "Transcription failed: \(error.localizedDescription)"
            }

            // Replace pending with correction result and promote
            if !correctionText.isEmpty {
                confirmedText += correctionText
                pendingText = ""
                confirmedSampleEnd = endPos
                fastChunkEnd = endPos
                transcriptionText = confirmedText
            }

            correctionTask = nil
        }
    }

    func stopRecording() {
        liveTask?.cancel()
        liveTask = nil
        correctionTask?.cancel()
        correctionTask = nil

        // Final correction: transcribe everything since last confirmed position
        let finalStart = confirmedSampleEnd
        let hasPending = recorder.sampleCount > finalStart

        _ = recorder.stopRecording()

        if hasPending, let (audio, _) = recorder.getAudio(from: finalStart) {
            generationTask = Task {
                guard let model = model else { return }

                isGenerating = true
                generationProgress = "Final transcription..."
                pendingText = ""

                var finalText = ""
                do {
                    for try await event in model.generateStream(
                        audio: audio,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        language: language,
                        chunkDuration: chunkDuration
                    ) {
                        try Task.checkCancellation()
                        switch event {
                        case .token(let token):
                            finalText += token
                            pendingText = finalText
                            transcriptionText = confirmedText + pendingText
                        case .info(let info):
                            tokensPerSecond = info.tokensPerSecond
                            peakMemory = info.peakMemoryUsage
                        case .result:
                            break
                        }
                    }
                } catch is CancellationError {
                    Memory.clearCache()
                } catch {
                    errorMessage = "Transcription failed: \(error.localizedDescription)"
                }

                if !finalText.isEmpty {
                    confirmedText += finalText
                    pendingText = ""
                    transcriptionText = confirmedText
                }

                generationProgress = ""
                isGenerating = false
            }
        } else {
            // Promote any remaining pending text
            confirmedText += pendingText
            pendingText = ""
            transcriptionText = confirmedText
        }
    }

    func cancelRecording() {
        liveTask?.cancel()
        liveTask = nil
        correctionTask?.cancel()
        correctionTask = nil
        recorder.cancelRecording()
        confirmedSampleEnd = 0
        fastChunkEnd = 0
    }


    func stop() {
        liveTask?.cancel()
        liveTask = nil
        correctionTask?.cancel()
        correctionTask = nil
        generationTask?.cancel()
        generationTask = nil

        if isRecording {
            recorder.cancelRecording()
            confirmedSampleEnd = 0
            fastChunkEnd = 0
        }

        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    func play() {
        audioPlayer.play()
    }

    func pause() {
        audioPlayer.pause()
    }

    func togglePlayPause() {
        audioPlayer.togglePlayPause()
    }

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }

    func copyTranscription() {
        #if os(iOS)
        UIPasteboard.general.string = transcriptionText
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcriptionText, forType: .string)
        #endif
    }

    private func resampleAudio(_ audio: MLXArray, from sourceSR: Int, to targetSR: Int) throws -> MLXArray {
        let samples = audio.asArray(Float.self)

        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sourceSR), channels: 1, interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(targetSR), channels: 1, interleaved: false
        ) else {
            throw NSError(domain: "STT", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio formats"])
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw NSError(domain: "STT", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"])
        }

        let inputFrameCount = AVAudioFrameCount(samples.count)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
            throw NSError(domain: "STT", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create input buffer"])
        }
        inputBuffer.frameLength = inputFrameCount
        memcpy(inputBuffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

        let ratio = Double(targetSR) / Double(sourceSR)
        let outputFrameCount = AVAudioFrameCount(Double(samples.count) * ratio)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
            throw NSError(domain: "STT", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create output buffer"])
        }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let error { throw error }

        let outputSamples = Array(UnsafeBufferPointer(
            start: outputBuffer.floatChannelData![0], count: Int(outputBuffer.frameLength)
        ))
        return MLXArray(outputSamples)
    }
}
