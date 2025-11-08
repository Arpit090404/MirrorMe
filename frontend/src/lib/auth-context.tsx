"use client"

import * as React from "react"

type User = {
  email?: string
  name?: string
  isGuest: boolean
}

type AuthContextType = {
  user: User | null
  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string, name: string) => Promise<void>
  loginAsGuest: () => void
  logout: () => void
}

const AuthContext = React.createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<User | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("mirrorme-user")
      return saved ? JSON.parse(saved) : null
    }
    return null
  })

  const login = async (email: string, password: string) => {
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 500))
    const newUser: User = { email, isGuest: false }
    setUser(newUser)
    if (typeof window !== "undefined") {
      localStorage.setItem("mirrorme-user", JSON.stringify(newUser))
    }
  }

  const signup = async (email: string, password: string, name: string) => {
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 500))
    const newUser: User = { email, name, isGuest: false }
    setUser(newUser)
    if (typeof window !== "undefined") {
      localStorage.setItem("mirrorme-user", JSON.stringify(newUser))
    }
  }

  const loginAsGuest = () => {
    const guestUser: User = { isGuest: true }
    setUser(guestUser)
    if (typeof window !== "undefined") {
      localStorage.setItem("mirrorme-user", JSON.stringify(guestUser))
    }
  }

  const logout = () => {
    setUser(null)
    if (typeof window !== "undefined") {
      localStorage.removeItem("mirrorme-user")
    }
  }

  return (
    <AuthContext.Provider value={{ user, login, signup, loginAsGuest, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = React.useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

