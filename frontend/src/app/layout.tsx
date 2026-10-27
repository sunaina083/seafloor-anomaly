import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Seafloor Anomaly Review",
  description: "Survey review tool for seafloor debris and anomaly detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}