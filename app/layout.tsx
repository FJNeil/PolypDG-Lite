import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "PolypDG-Lite | Lightweight Domain Generalization Research", description: "Research project page for PolypDG-Lite: a lightweight domain-generalization study for robust polyp image analysis." };
export default function RootLayout({children}: Readonly<{children:React.ReactNode}>) { return <html lang="en"><body>{children}</body></html>; }
