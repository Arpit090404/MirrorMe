"use client"

import * as React from "react"
import { Video, VideoOff, Mic, MicOff } from "lucide-react"
import { Button } from "@/components/ui/button"

interface VideoStreamProps {
  isRecording: boolean
  onCameraToggle: (enabled: boolean) => void
  onMicToggle: (enabled: boolean) => void
  onFrameCapture?: (canvas: HTMLCanvasElement) => void
}

export function VideoStream({ isRecording, onCameraToggle, onMicToggle, onFrameCapture }: VideoStreamProps) {
  const videoRef = React.useRef<HTMLVideoElement>(null)
  const streamRef = React.useRef<MediaStream | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [cameraEnabled, setCameraEnabled] = React.useState(false)
  const [micEnabled, setMicEnabled] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  React.useEffect(() => {
    if (isRecording && !cameraEnabled) {
      // Auto-start camera when recording begins
      startCamera()
    } else if (!isRecording && streamRef.current) {
      // Stop all tracks when recording stops
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
      setCameraEnabled(false)
      setMicEnabled(false)
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    }
  }, [isRecording])

  // Capture frames for backend processing
  React.useEffect(() => {
    if (!isRecording || !cameraEnabled) return

    const captureFrame = () => {
      if (videoRef.current && canvasRef.current && onFrameCapture) {
        const video = videoRef.current
        const canvas = canvasRef.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.drawImage(video, 0, 0)
          onFrameCapture(canvas)
        }
      }
    }

    const interval = setInterval(captureFrame, 500)
    return () => clearInterval(interval)
  }, [isRecording, cameraEnabled, onFrameCapture])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 1280, height: 720 },
        audio: true
      })
      
      streamRef.current = stream
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      
      setCameraEnabled(true)
      setMicEnabled(true)
      setError(null)
      onCameraToggle(true)
      onMicToggle(true)
    } catch (err) {
      console.error("Error accessing camera/microphone:", err)
      setError("Could not access camera/microphone. Please grant permissions.")
      onCameraToggle(false)
      onMicToggle(false)
    }
  }

  const toggleCamera = () => {
    if (streamRef.current) {
      const videoTracks = streamRef.current.getVideoTracks()
      videoTracks.forEach(track => {
        track.enabled = !cameraEnabled
      })
      setCameraEnabled(!cameraEnabled)
      onCameraToggle(!cameraEnabled)
    }
  }

  const toggleMic = () => {
    if (streamRef.current) {
      const audioTracks = streamRef.current.getAudioTracks()
      audioTracks.forEach(track => {
        track.enabled = !micEnabled
      })
      setMicEnabled(!micEnabled)
      onMicToggle(!micEnabled)
    }
  }

  return (
    <div className="relative">
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <div className="aspect-video bg-gradient-to-br from-slate-200 to-slate-300 dark:from-slate-800 dark:to-slate-900 rounded-xl flex items-center justify-center overflow-hidden relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted={!micEnabled}
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />
        
        {!cameraEnabled && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-slate-300 to-slate-400 dark:from-slate-700 dark:to-slate-800">
            <VideoOff className="w-16 h-16 mb-4 text-muted-foreground" />
            <p className="text-muted-foreground text-center px-4">
              {error || "Camera feed will appear here"}
            </p>
          </div>
        )}

        {cameraEnabled && isRecording && (
          <div className="absolute top-2 right-2">
            <div className="bg-red-500 text-white px-2 py-1 rounded-full text-xs font-semibold flex items-center gap-1 shadow-lg">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              LIVE
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      {isRecording && (
        <div className="mt-3 flex gap-2">
          <Button
            variant={cameraEnabled ? "default" : "outline"}
            onClick={cameraEnabled ? toggleCamera : startCamera}
            className="flex-1"
          >
            {cameraEnabled ? (
              <>
                <VideoOff className="w-4 h-4 mr-2" />
                Camera Off
              </>
            ) : (
              <>
                <Video className="w-4 h-4 mr-2" />
                Camera On
              </>
            )}
          </Button>
          <Button
            variant={micEnabled ? "default" : "outline"}
            onClick={micEnabled ? toggleMic : startCamera}
            disabled={!cameraEnabled}
            className="flex-1"
          >
            {micEnabled ? (
              <>
                <MicOff className="w-4 h-4 mr-2" />
                Mic Off
              </>
            ) : (
              <>
                <Mic className="w-4 h-4 mr-2" />
                Mic On
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  )
}

