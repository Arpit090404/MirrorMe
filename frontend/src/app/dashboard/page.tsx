"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { LogOut, Video, Mic, TrendingUp, Eye, Smile, User, ArrowLeft, Settings } from "lucide-react"
import axios from "axios"
import { VideoStream } from "@/components/video-stream"

export default function DashboardPage() {
  const { user, logout } = useAuth()
  const router = useRouter()
  const [isRecording, setIsRecording] = React.useState(false)
  const [prediction, setPrediction] = React.useState("Starting...")
  const [scores, setScores] = React.useState({
    blink: 0,
    head_stability: 0,
    gesture_activity: 0,
    eye_contact: 0,
    voice: 0,
  })
  const [isProcessing, setIsProcessing] = React.useState(false)
  const [mounted, setMounted] = React.useState(false)
  const [cameraEnabled, setCameraEnabled] = React.useState(false)
  const [micEnabled, setMicEnabled] = React.useState(false)
  const [showSummary, setShowSummary] = React.useState(false)
  const [isAnalyzing, setIsAnalyzing] = React.useState(false)
  const [sessionData, setSessionData] = React.useState<any[]>([])
  const intervalRef = React.useRef<NodeJS.Timeout | null>(null)
  const sessionDataRef = React.useRef<any[]>([])
  const isRecordingRef = React.useRef(false) // Track recording state via ref to avoid stale closures

  React.useEffect(() => {
    setMounted(true)
  }, [])

  // Cleanup interval on unmount
  React.useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  // Redirect to login if not authenticated (must be in useEffect, not during render)
  React.useEffect(() => {
    if (mounted && !user) {
      router.push("/login")
    }
  }, [mounted, user, router])

  // Calculate average scores for summary - MUST be before any early returns
  const avgScores = React.useMemo(() => {
    if (sessionData.length === 0) return {}
    
    const scores = sessionData.reduce((acc, data) => {
      if (data.scores && typeof data.scores === 'object') {
        Object.keys(data.scores).forEach(key => {
          const value = data.scores[key]
          if (typeof value === 'number' && !isNaN(value)) {
            acc[key] = (acc[key] || 0) + value
            acc[`${key}_count`] = (acc[`${key}_count`] || 0) + 1
          }
        })
      }
      return acc
    }, {} as any)
    
    const averaged: any = {}
    Object.keys(scores).forEach(key => {
      if (!key.endsWith('_count')) {
        const countKey = `${key}_count`
        const count = scores[countKey] || 1
        averaged[key] = Math.round(scores[key] / count)
      }
    })
    
    return averaged
  }, [sessionData])
  
  // Also calculate prediction distribution - MUST be before any early returns
  const predictionDistribution = React.useMemo(() => {
    if (sessionData.length === 0) return { Confident: 0, Nervous: 0, Neutral: 0 }
    
    const dist = { Confident: 0, Nervous: 0, Neutral: 0 }
    sessionData.forEach(data => {
      const pred = data.prediction || "Neutral"
      if (pred in dist) {
        dist[pred as keyof typeof dist]++
      }
    })
    return dist
  }, [sessionData])
  
  // Most common prediction - MUST be before any early returns
  const mostCommonPrediction = React.useMemo(() => {
    const dist = predictionDistribution
    return Object.entries(dist).reduce((a, b) => dist[a[0] as keyof typeof dist] > dist[b[0] as keyof typeof dist] ? a : b)[0]
  }, [predictionDistribution])
  
  // Calculate overall score from current scores (real-time) or average (summary) - MUST be before any early returns
  const currentOverallScore = React.useMemo(() => {
    return Object.keys(scores).length > 0
      ? Math.round(Object.values(scores).reduce((a: any, b: any) => a + b, 0) / Object.keys(scores).length)
      : 0
  }, [scores])
  
  // Calculate overall score - use average if available, otherwise current - MUST be before any early returns
  const overallScore = React.useMemo(() => {
    if (sessionData.length > 0 && Object.keys(avgScores).length > 0) {
      const avg = Object.values(avgScores).reduce((a: any, b: any) => a + b, 0) / Object.keys(avgScores).length
      return Math.round(avg)
    }
    return currentOverallScore
  }, [sessionData, avgScores, currentOverallScore])

  const handleStart = async () => {
    setIsRecording(true)
    isRecordingRef.current = true // Update ref immediately
    setIsProcessing(true)
    setPrediction("Starting...")
    setScores({
      blink: 0,
      head_stability: 0,
      gesture_activity: 0,
      eye_contact: 0,
      voice: 0,
    })
    setShowSummary(false)
    setIsAnalyzing(false)
    setSessionData([])
    sessionDataRef.current = []
    
    try {
      await axios.post("http://127.0.0.1:5000/start")
      setIsProcessing(false)
      
      // Poll for updates (only while recording)
      intervalRef.current = setInterval(async () => {
        // Check recording state via ref (avoids stale closure)
        if (!isRecordingRef.current) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          return
        }
        
        try {
          const response = await axios.get("http://127.0.0.1:5000/process")
          console.log("Response received:", response.data)
          
          // Double-check we're still recording before updating (defensive programming)
          if (isRecordingRef.current) {
            setPrediction(response.data.prediction || "Neutral")
            if (response.data.scores) {
              console.log("Scores:", response.data.scores)
              setScores(response.data.scores)
              // Store session data with timestamp for better tracking
              const dataPoint = {
                ...response.data,
                timestamp: Date.now()
              }
              setSessionData(prev => {
                const updated = [...prev, dataPoint]
                // Keep last 100 data points (about 50 seconds of data at 500ms intervals)
                const sliced = updated.slice(-100)
                sessionDataRef.current = sliced // Keep ref in sync
                return sliced
              })
            }
          }
        } catch (error) {
          console.error("Error fetching data:", error)
        }
      }, 500)
    } catch (error) {
      console.error("Error starting recording:", error)
      isRecordingRef.current = false
      setIsRecording(false)
      setIsProcessing(false)
    }
  }

  const handleStop = async () => {
    // Immediately stop recording (both state and ref) and clear interval
    isRecordingRef.current = false // Set ref first to stop any pending callbacks
    setIsRecording(false)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    
    // Freeze current values - don't update prediction/scores anymore
    // The current prediction and scores will remain as the last values during recording
    
    try {
      await axios.post("http://127.0.0.1:5000/stop")
      // Show analyzing state
      setIsAnalyzing(true)
      // Wait a bit for any pending data to be collected, then show summary
      setTimeout(() => {
        // Use ref to get latest sessionData (bypasses closure issue)
        const currentData = sessionDataRef.current
        console.log("Session data length:", currentData.length)
        console.log("Session data:", currentData)
        setIsAnalyzing(false)
        // Always show summary after stopping
        setShowSummary(true)
      }, 1500) // Give time for final data processing
    } catch (error) {
      console.error("Error stopping recording:", error)
      setIsAnalyzing(false)
      // Show summary anyway
      setTimeout(() => setShowSummary(true), 500)
    }
  }

  const handleLogout = () => {
    logout()
    router.push("/login")
  }

  if (!mounted) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>
  }
  
  if (!user) {
    return null // useEffect will handle the redirect
  }

  const predictionColors = {
    Confident: "text-green-500",
    Nervous: "text-orange-500",
    Neutral: "text-blue-500",
  }
  
  // Dynamic tips based on current scores
  const getImprovementTips = () => {
    const tips = []
    if (scores.eye_contact < 70) {
      tips.push("Look directly at the camera lens - maintain steady eye contact")
    }
    if (scores.head_stability < 70) {
      tips.push("Keep your head steady - minimize side-to-side movement")
    }
    if (scores.gesture_activity < 60) {
      tips.push("Use natural hand gestures to emphasize key points")
    }
    if (scores.voice < 70) {
      tips.push("Speak clearly at a comfortable volume - project your voice")
    }
    if (scores.blink < 70) {
      tips.push("Blink naturally - too frequent blinking may indicate nervousness")
    }
    if (tips.length === 0) {
      tips.push("Excellent! Maintain your confident posture")
      tips.push("Keep using natural gestures to engage")
      tips.push("Your eye contact and voice are strong")
    }
    return tips.slice(0, 4) // Max 4 tips
  }

  // Detailed feedback after analysis based on all metrics
  const getDetailedFeedback = () => {
    const feedback: { category: string; score: number; tips: string[]; priority: number }[] = []
    
    // Use average scores if available, otherwise use current scores
    const scoresToUse = Object.keys(avgScores).length > 0 ? avgScores : scores
    
    // Eye Contact
    const eyeScore = scoresToUse.eye_contact || 0
    const eyeTips: string[] = []
    if (eyeScore < 70) {
      eyeTips.push("Look directly at the camera lens, not at the screen")
      eyeTips.push("Practice maintaining eye contact for 3-5 seconds at a time")
      if (eyeScore < 50) {
        eyeTips.push("Avoid looking down or away frequently")
      }
    } else {
      eyeTips.push("Great eye contact! Keep this up")
    }
    feedback.push({ category: "Eye Contact", score: eyeScore, tips: eyeTips, priority: eyeScore < 70 ? 1 : 3 })
    
    // Head Stability
    const headScore = scoresToUse.head_stability || 0
    const headTips: string[] = []
    if (headScore < 70) {
      headTips.push("Keep your head centered and minimize unnecessary movement")
      headTips.push("Practice speaking with a steady, confident posture")
      if (headScore < 50) {
        headTips.push("Avoid rapid head movements - they can indicate nervousness")
      }
    } else {
      headTips.push("Excellent head stability! You look confident")
    }
    feedback.push({ category: "Head Stability", score: headScore, tips: headTips, priority: headScore < 70 ? 1 : 3 })
    
    // Gesture Activity
    const gestureScore = scoresToUse.gesture_activity || 0
    const gestureTips: string[] = []
    if (gestureScore < 60) {
      gestureTips.push("Use natural hand movements to emphasize your points")
      gestureTips.push("Open gestures (palms visible) show confidence")
      if (gestureScore < 40) {
        gestureTips.push("Avoid keeping your hands hidden or still")
      }
    } else {
      gestureTips.push("Good use of gestures! They make you more engaging")
    }
    feedback.push({ category: "Gesture Activity", score: gestureScore, tips: gestureTips, priority: gestureScore < 60 ? 2 : 3 })
    
    // Voice
    const voiceScore = scoresToUse.voice || 0
    const voiceTips: string[] = []
    if (voiceScore < 70) {
      voiceTips.push("Speak at a comfortable volume - not too quiet or loud")
      voiceTips.push("Maintain consistent volume throughout your speech")
      if (voiceScore < 50) {
        voiceTips.push("Practice speaking clearly and avoid mumbling")
        voiceTips.push("Reduce hesitation pauses (um, uh) in your speech")
      }
    } else {
      voiceTips.push("Great voice projection! Clear and consistent")
    }
    feedback.push({ category: "Voice", score: voiceScore, tips: voiceTips, priority: voiceScore < 70 ? 1 : 3 })
    
    // Blink Rate
    const blinkScore = scoresToUse.blink || 0
    const blinkTips: string[] = []
    if (blinkScore < 70) {
      blinkTips.push("Blink naturally - aim for 10-15 blinks per minute")
      if (blinkScore < 50) {
        blinkTips.push("Frequent blinking can indicate stress - try to relax")
      }
    } else {
      blinkTips.push("Natural blinking pattern - well done")
    }
    feedback.push({ category: "Blink Rate", score: blinkScore, tips: blinkTips, priority: 2 })
    
    // Sort by priority (lower score = higher priority)
    return feedback.sort((a, b) => a.priority - b.priority)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Header */}
      <header className="glass border-b border-border/50 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => router.push("/")}
              className="rounded-full"
            >
              <ArrowLeft className="w-4 h-4" />
            </Button>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <Video className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">MirrorMe</h1>
              <p className="text-xs text-muted-foreground">
                {user.isGuest ? "Guest Mode" : user.email}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            {!user.isGuest && (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleLogout}
                className="rounded-full"
              >
                <LogOut className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 max-w-7xl">
        {/* Video Feed Section - Full Width */}
        <Card className="glass-strong p-6 mb-6">
          <div className="grid md:grid-cols-3 gap-6">
            {/* Video Feed - Left Side */}
            <div className="md:col-span-2">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Video className="w-5 h-5" />
                Live Feed
              </h2>
              
              <VideoStream
                isRecording={isRecording}
                onCameraToggle={setCameraEnabled}
                onMicToggle={setMicEnabled}
                onFrameCapture={async (canvas) => {
                  // Send frame to backend
                  const dataUrl = canvas.toDataURL('image/jpeg', 0.8)
                  try {
                    await axios.post("http://127.0.0.1:5000/process_frame", { frame: dataUrl })
                  } catch (error) {
                    console.error("Error sending frame:", error)
                  }
                }}
              />

              <div className="flex gap-3 mt-4">
                <Button
                  onClick={isRecording ? handleStop : handleStart}
                  disabled={isProcessing}
                  className="flex-1 h-12 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white font-semibold shadow-lg"
                >
                  {isRecording ? (
                    <>
                      <Mic className="w-4 h-4 mr-2" />
                      Stop Analysis
                    </>
                  ) : (
                    <>
                      <Video className="w-4 h-4 mr-2" />
                      Start Analysis
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Real-Time Analysis - Right Side */}
            <div className="md:col-span-1 space-y-4">
              <div>
                <h2 className="text-xl font-semibold mb-4">Real-Time Analysis</h2>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className={`text-5xl font-bold mb-2 ${predictionColors[prediction as keyof typeof predictionColors] || predictionColors.Neutral}`}>
                      {prediction}
                    </div>
                    <div className="text-sm text-muted-foreground mb-4">
                      Current mood assessment
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Overall Score</span>
                        <span className="font-semibold">{currentOverallScore}%</span>
                      </div>
                      <Progress 
                        value={currentOverallScore} 
                        className="h-3"
                      />
                    </div>
                    {isRecording && (
                      <div className="mt-4 pt-4 border-t border-border/50">
                        <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
                          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                          <span>Live Analysis Active</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Real-Time Tips - Right Below Analysis */}
              <Card className="glass-strong p-4">
                <h2 className="text-lg font-semibold mb-3">Real-Time Tips</h2>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  {getImprovementTips().map((tip, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-primary mt-0.5">•</span>
                      <span>{tip}</span>
                    </li>
                  ))}
                </ul>
                {isRecording && (
                  <div className="mt-3 pt-3 border-t border-border/50">
                    <div className="text-xs text-muted-foreground">
                      Updates every 500ms
                    </div>
                  </div>
                )}
              </Card>
            </div>
          </div>
        </Card>

        {/* Metrics and Summary - Full Width */}
        <div className="space-y-6">
          {/* Performance Metrics - Full Width */}
          <div>
            <Card className="glass-strong p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Performance Metrics
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {/* Blink Rate */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Eye className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Blink Rate</span>
                    </div>
                    <span className="text-sm font-semibold">{scores.blink}%</span>
                  </div>
                  <Progress value={scores.blink} className="h-2" />
                </div>

                {/* Head Stability */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Head Stability</span>
                    </div>
                    <span className="text-sm font-semibold">{scores.head_stability}%</span>
                  </div>
                  <Progress value={scores.head_stability} className="h-2" />
                </div>

                {/* Eye Contact */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Eye className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Eye Contact</span>
                    </div>
                    <span className="text-sm font-semibold">{scores.eye_contact}%</span>
                  </div>
                  <Progress value={scores.eye_contact} className="h-2" />
                </div>

                {/* Gesture Activity */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Smile className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Gestures</span>
                    </div>
                    <span className="text-sm font-semibold">{scores.gesture_activity}%</span>
                  </div>
                  <Progress value={scores.gesture_activity} className="h-2" />
                </div>

                {/* Voice */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Mic className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Voice Volume</span>
                    </div>
                    <span className="text-sm font-semibold">{scores.voice}%</span>
                  </div>
                  <Progress value={scores.voice} className="h-2" />
                </div>
              </div>
            </Card>
          </div>

          {/* Analyzing State - Shown after stopping, before summary */}
          {isAnalyzing && (
            <Card className="glass-strong p-6 border-2 border-primary/30">
              <div className="flex flex-col items-center justify-center py-8">
                <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
                <h2 className="text-xl font-semibold mb-2">Analyzing Your Performance...</h2>
                <p className="text-sm text-muted-foreground text-center">
                  Processing {sessionData.length} frames and calculating your session summary
                </p>
              </div>
            </Card>
          )}

          {/* Session Summary Card - Only shown after analyzing */}
          {showSummary && !isAnalyzing && (
            <Card className="glass-strong p-6 border-2 border-primary/30">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  Session Summary
                </h2>
                <div className="space-y-6">
                  <div className="bg-gradient-to-r from-primary/10 to-purple-500/10 rounded-xl p-4">
                    <div className="text-sm text-muted-foreground mb-1">Overall Score</div>
                    <div className="text-4xl font-bold text-primary">{overallScore}%</div>
                    <div className="text-xs text-muted-foreground mt-2">
                      {sessionData.length > 0 
                        ? `Based on ${sessionData.length} analyzed frames`
                        : `Based on current analysis`}
                    </div>
                    {sessionData.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-primary/20">
                        <div className="text-xs text-muted-foreground">Most Common Prediction:</div>
                        <div className={`text-sm font-semibold mt-1 ${predictionColors[mostCommonPrediction as keyof typeof predictionColors] || predictionColors.Neutral}`}>
                          {mostCommonPrediction}
                        </div>
                        <div className="text-xs text-muted-foreground mt-2">
                          Distribution: {Object.entries(predictionDistribution).map(([k, v]) => `${k}: ${Math.round((v / sessionData.length) * 100)}%`).join(', ')}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-semibold mb-3">
                      {sessionData.length > 0 ? 'Average Scores' : 'Current Scores'}
                    </h3>
                    <div className="space-y-2">
                      {(Object.keys(avgScores).length > 0 ? Object.entries(avgScores) : Object.entries(scores)).map(([key, value]: [string, any]) => (
                        <div key={key}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium capitalize">{key.replace(/_/g, ' ')}</span>
                            <span className="text-sm font-semibold">{Math.round(value)}%</span>
                          </div>
                          <Progress value={Math.round(value)} className="h-1.5" />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold mb-3">Detailed Feedback & Improvements</h3>
                    <div className="space-y-4">
                      {getDetailedFeedback().slice(0, 5).map((item, idx) => (
                        <div key={idx} className="p-3 rounded-lg bg-muted/50">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">{item.category}</span>
                            <span className={`text-xs font-semibold ${item.score >= 70 ? 'text-green-600' : item.score >= 50 ? 'text-yellow-600' : 'text-red-600'}`}>
                              {item.score}%
                            </span>
                          </div>
                          <ul className="text-xs text-muted-foreground space-y-1 ml-4">
                            {item.tips.map((tip, tipIdx) => (
                              <li key={tipIdx} className="list-disc">{tip}</li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* How to Get Confident Section */}
                  {overallScore < 75 && (
                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                      <h3 className="text-sm font-semibold mb-2 text-blue-600 dark:text-blue-400">
                        💡 How to Get "Confident" Prediction
                      </h3>
                      <ul className="text-xs text-muted-foreground space-y-1">
                        <li>• Aim for 75%+ in all metrics (eye contact, head stability, voice, gestures)</li>
                        <li>• Maintain steady eye contact with camera lens (not screen)</li>
                        <li>• Keep head centered with minimal side-to-side movement</li>
                        <li>• Speak clearly at comfortable volume with minimal hesitation</li>
                        <li>• Use natural hand gestures to emphasize points</li>
                        <li>• Practice regularly - confidence builds with repetition!</li>
                      </ul>
                      <div className="mt-2 text-xs text-muted-foreground">
                        <strong>Current Overall:</strong> {overallScore}% - Target: 75%+
                      </div>
                    </div>
                  )}
                  
                  <Button
                    onClick={() => setShowSummary(false)}
                    variant="outline"
                    className="w-full"
                  >
                    Close Summary
                  </Button>
                </div>
              </Card>
            )}
        </div>
      </main>
    </div>
  )
}

