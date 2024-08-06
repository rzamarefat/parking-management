import React, { useEffect, useRef } from 'react';
import io from 'socket.io-client';

const VideoStream = () => {
  const canvasRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = io('http://localhost:5000');

    socketRef.current.on('frame', (frameData) => {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      const img = new Image();
      img.src = 'data:image/jpeg;base64,' + btoa(String.fromCharCode(...new Uint8Array(frameData)));
      img.onload = () => {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
    });

    
    return () => {
      socketRef.current.disconnect();
    };
  }, []);
  
  return <canvas ref={canvasRef} width="640" height="480"></canvas>;
};

export default VideoStream;
